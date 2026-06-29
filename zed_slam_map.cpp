// zed_slam_map.cpp — Underwater Hybrid SLAM
//                    *** UNDERWATER VERSION — FULL HYBRID APPROACH ***
//
//  Root problem solved here:
//    The ZED SDK's internal spatial mapping uses the factory AIR calibration for
//    stereo matching.  Underwater (flat-port, n_water/n_air ≈ 1.33), the focal
//    length is ~25 % larger and the disparity scale is completely different →
//    the SDK's depth is mostly NaN → spatial mapping accumulates NOTHING.
//
//  Solution — Hybrid SLAM:
//    • ZED SDK positional tracking (visual odometry) → camera pose per frame.
//      Works with DEPTH_MODE::NONE — only needs 2-D feature matching, no depth.
//    • OpenCV SGBM with underwater_calib.yaml Q matrix → correct depth per frame.
//      Same preprocessing pipeline as zed_rviz.cpp.
//    • Per-frame: transform SGBM point cloud to world frame using SDK pose.
//    • Accumulate in a voxel map (4 cm cells, up to 300k voxels).
//    • Publish accumulated map as PointCloud2 + RViz Marker (POINTS).
//    • GLFW: live coloured point-cloud preview of the growing world map.
//
//  Coordinate systems:
//    SGBM (OpenCV pinhole): X=right, Y=down, Z=forward (depth)
//    ZED SDK camera frame (RIGHT_HANDED_Y_UP): X=right, Y=up, Z=backward
//    ZED SDK world frame (RIGHT_HANDED_Y_UP): accumulated from start pose
//    ROS REP-103: X=forward, Y=left, Z=up
//
//  Controls (GLFW window):
//    Left-drag → orbit      Scroll → zoom

#include <sl/Camera.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <yaml-cpp/yaml.h>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GLFW/glfw3.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>

#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <iostream>
#include <filesystem>

using namespace sl;

// ─── YAML helper ─────────────────────────────────────────────────────────────
static cv::Mat yamlToMat(const YAML::Node& node, int rows, int cols, int type)
{
    cv::Mat m(rows, cols, type);
    int k = 0;
    for (const auto& v : node)
        m.at<double>(k / cols, k % cols) = v.as<double>(), ++k;
    return m;
}

// ─── Underwater preprocessing (percentile stretch + CLAHE + unsharp mask) ────
static void underwaterPreprocess(const cv::Mat& bgr_in, cv::Mat& bgr_out,
                                  cv::Ptr<cv::CLAHE>& clahe)
{
    std::vector<cv::Mat> ch(3);
    cv::split(bgr_in, ch);
    for (int i = 0; i < 3; ++i) {
        cv::Mat f32; ch[i].convertTo(f32, CV_32F);
        std::vector<float> v(f32.begin<float>(), f32.end<float>());
        std::nth_element(v.begin(), v.begin() + int(0.01f*v.size()), v.end());
        float lo = v[int(0.01f*v.size())];
        std::nth_element(v.begin(), v.begin() + int(0.99f*v.size()), v.end());
        float hi = v[int(0.99f*v.size())];
        if (hi <= lo) continue;
        f32 = (f32 - lo) * (255.f / (hi - lo));
        f32 = cv::max(f32, 0.f); f32 = cv::min(f32, 255.f);
        f32.convertTo(ch[i], CV_8U);
        clahe->apply(ch[i], ch[i]);
    }
    cv::merge(ch, bgr_out);
    cv::Mat blur;
    cv::GaussianBlur(bgr_out, blur, cv::Size(0,0), 1.5);
    cv::addWeighted(bgr_out, 1.4, blur, -0.4, 0.0, bgr_out);
}

// ─── Transform SGBM camera-space point → ZED world-space ─────────────────────
//
//  SGBM (OpenCV pinhole convention):  X=right, Y=down,     Z=forward (depth)
//  ZED SDK camera frame (RH_Y_UP):    X=right, Y=up,       Z=backward
//  Conversion: cx =  px, cy = -py, cz = -pz
//
//  pose_data is a 4×4 row-major matrix that transforms ZED camera → world frame.
//
inline void sgbmToWorld(float px, float py, float pz,
                        const sl::Transform& tf,
                        float& wx, float& wy, float& wz)
{
    // Step 1: SGBM → ZED camera frame
    float cx =  px;
    float cy = -py;
    float cz = -pz;

    // Step 2: ZED camera frame → world frame  (row-major 4×4)
    const float* M = tf.m;
    wx = M[0]*cx + M[1]*cy + M[ 2]*cz + M[ 3];
    wy = M[4]*cx + M[5]*cy + M[ 6]*cz + M[ 7];
    wz = M[8]*cx + M[9]*cy + M[10]*cz + M[11];
}

// ─── ROS REP-103 frame from ZED RIGHT_HANDED_Y_UP world frame ────────────────
inline void toROS(float wx, float wy, float wz,
                  float& rx, float& ry, float& rz)
{
    rx = -wz;   // ROS X (forward)  = -ZED Z (backward)
    ry = -wx;   // ROS Y (left)     = -ZED X (right)
    rz =  wy;   // ROS Z (up)       =  ZED Y (up)
}

// ─── Voxel map — accumulated world-space point cloud ─────────────────────────
//  Each 4 cm cell stores the most recent observed (x,y,z,r,g,b).
//  Insertion is O(1) average; memory is bounded by MAX_VOXELS.
struct VoxelMap {
    static constexpr float  CELL       = 0.04f;     // 4 cm per voxel
    static constexpr size_t MAX_VOXELS = 300'000;   // ~7 MB of voxel data

    struct Key {
        int x, y, z;
        bool operator==(const Key& o) const {
            return x==o.x && y==o.y && z==o.z;
        }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            return (size_t)k.x*73856093ULL
                 ^ (size_t)k.y*19349663ULL
                 ^ (size_t)k.z*83492791ULL;
        }
    };
    struct Voxel { float wx,wy,wz, r,g,b; };

    std::unordered_map<Key, Voxel, KeyHash> cells;

    // Insert a world-space point.  Newest colour for each cell wins.
    // Returns false if the map is full (so we stop trying this frame).
    bool insert(float wx, float wy, float wz, float r, float g, float b)
    {
        if (cells.size() >= MAX_VOXELS) return false;
        Key k{ (int)std::floor(wx / CELL),
               (int)std::floor(wy / CELL),
               (int)std::floor(wz / CELL) };
        cells[k] = { wx, wy, wz, r, g, b };
        return true;
    }

    // Dump to flat [x,y,z,r,g,b] vector for GL / ROS
    void toVector(std::vector<float>& out) const
    {
        out.clear();
        out.reserve(cells.size() * 6);
        for (const auto& [k, v] : cells) {
            out.push_back(v.wx); out.push_back(v.wy); out.push_back(v.wz);
            out.push_back(v.r);  out.push_back(v.g);  out.push_back(v.b);
        }
    }

    size_t size() const { return cells.size(); }
    void   clear()      { cells.clear(); }
};

// ─── Camera orbit ─────────────────────────────────────────────────────────────
struct Camera3D {
    float yaw=0.f, pitch=-25.f, dist=8.f;   // dist=8 to see wider world map
    float cx=0, cy=0, cz=0;
    double last_mx=-1, last_my=-1;
    bool dragging=false;
} g_cam;

void scroll_cb(GLFWwindow*,double,double dy)
    { g_cam.dist-=(float)dy*0.3f; g_cam.dist=std::max(0.5f,g_cam.dist); }
void mouse_btn_cb(GLFWwindow*,int btn,int action,int)
    { if(btn==GLFW_MOUSE_BUTTON_LEFT) g_cam.dragging=(action==GLFW_PRESS); }
void cursor_cb(GLFWwindow*,double x,double y){
    if(g_cam.dragging&&g_cam.last_mx>=0){
        g_cam.yaw  +=(float)(x-g_cam.last_mx)*0.4f;
        g_cam.pitch+=(float)(y-g_cam.last_my)*0.4f;
        g_cam.pitch=std::max(-89.f,std::min(89.f,g_cam.pitch));
    }
    g_cam.last_mx=x; g_cam.last_my=y;
}

// ─── Shared point data (ROS thread → GL thread) ───────────────────────────────
// Flat [x,y,z,r,g,b] per voxel — ZED world frame (for GLFW)
struct PointData {
    std::vector<float> pts;
    std::atomic<bool>  fresh{false};
    std::mutex         mtx;
} g_pts;

// ─── OpenGL render — coloured point cloud ────────────────────────────────────
void drawPoints(const std::vector<float>& pts)
{
    if (pts.empty()) return;
    glPointSize(3.0f);
    glBegin(GL_POINTS);
    for (size_t i = 0; i + 5 < pts.size(); i += 6) {
        glColor3f(pts[i+3], pts[i+4], pts[i+5]);
        glVertex3f(pts[i],   pts[i+1], pts[i+2]);
    }
    glEnd();
}

// ─── ROS2 node ────────────────────────────────────────────────────────────────
class ZedSlamNode : public rclcpp::Node {
public:
    bool init_ok = false;

    ZedSlamNode() : Node("zed_slam_map")
    {
        // ── Publishers ───────────────────────────────────────────────────────
        // Accumulated world-space map → RViz PointCloud2
        pc_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            "zed/point_cloud", 10);
        // Same map as Marker POINTS → backward compat with existing RViz configs
        marker_pub_ = create_publisher<visualization_msgs::msg::Marker>(
            "zed/slam_mesh", 10);

        // ── Load underwater_calib.yaml ───────────────────────────────────────
        const std::string YAML_PATH =
            "/home/uday/ros2_ws/src/zed_rviz/scripts/underwater_calib.yaml";

        if (!std::filesystem::exists(YAML_PATH)) {
            RCLCPP_FATAL(get_logger(),
                "Underwater calibration YAML not found: %s", YAML_PATH.c_str());
            return;
        }

        YAML::Node cfg;
        try { cfg = YAML::LoadFile(YAML_PATH); }
        catch (const std::exception& e) {
            RCLCPP_FATAL(get_logger(), "YAML parse error: %s", e.what());
            return;
        }

        YAML::Node uw = cfg["underwater"];
        if (!uw) {
            RCLCPP_FATAL(get_logger(), "YAML missing 'underwater' section");
            return;
        }

        // ── Intrinsics + distortion ──────────────────────────────────────────
        cv::Mat K1 = yamlToMat(uw["left"]["K"],  3, 3, CV_64F);
        cv::Mat K2 = yamlToMat(uw["right"]["K"], 3, 3, CV_64F);
        cv::Mat D1 = yamlToMat(uw["left"]["D"],  1, 8, CV_64F).colRange(0,5);
        cv::Mat D2 = yamlToMat(uw["right"]["D"], 1, 8, CV_64F).colRange(0,5);

        // ── Rectification outputs ────────────────────────────────────────────
        cv::Mat R1 = yamlToMat(uw["R1"], 3, 3, CV_64F);
        cv::Mat R2 = yamlToMat(uw["R2"], 3, 3, CV_64F);
        cv::Mat P1 = yamlToMat(uw["P1"], 3, 4, CV_64F);
        cv::Mat P2 = yamlToMat(uw["P2"], 3, 4, CV_64F);
        Q_         = yamlToMat(uw["Q"],  4, 4, CV_64F);

        RCLCPP_INFO(get_logger(),
            "Underwater calib loaded: fx_L=%.2f  baseline=%.4f m",
            K1.at<double>(0,0), uw["baseline_m"].as<double>());

        // ── Build rectification maps ─────────────────────────────────────────
        const cv::Size IMG_SZ(1280, 720);
        cv::initUndistortRectifyMap(K1,D1,R1,P1, IMG_SZ, CV_32FC1, m1x_,m1y_);
        cv::initUndistortRectifyMap(K2,D2,R2,P2, IMG_SZ, CV_32FC1, m2x_,m2y_);

        // ── SGBM stereo matcher — same params as zed_rviz.cpp ───────────────
        int bs=5, nd=128;
        lm_ = cv::StereoSGBM::create(0, nd, bs,
            8*bs*bs, 32*bs*bs, 1, 0, 8, 60, 2,
            cv::StereoSGBM::MODE_HH);
        rm_  = cv::ximgproc::createRightMatcher(lm_);
        wls_ = cv::ximgproc::createDisparityWLSFilter(lm_);
        wls_->setLambda(8000);
        wls_->setSigmaColor(1.5);

        // ── CLAHE (clipLimit=2.5, tile=8×8) ─────────────────────────────────
        clahe_ = cv::createCLAHE(2.5, cv::Size(8, 8));

        // ── Open ZED — DEPTH_MODE::PERFORMANCE (minimum for positional tracking) ─
        //   NOTE: DEPTH_MODE::NONE disables the internal pose/IMU pipeline that
        //   enablePositionalTracking() depends on → causes the misleading SDK error
        //   "sl::Camera::Open has not been called, no Camera instance running."
        //   even though the camera opened fine.
        //
        //   Fix: use DEPTH_MODE::PERFORMANCE.  The SDK depth output is NEVER
        //   retrieved here — our SGBM pipeline does all depth work — so there is
        //   zero impact on accuracy.  The GPU overhead of PERFORMANCE mode is
        //   ~5-8 ms/frame, which is acceptable.
        InitParameters ip;
        ip.camera_resolution  = RESOLUTION::HD720;
        ip.camera_fps         = 30;
        ip.depth_mode         = DEPTH_MODE::PERFORMANCE;  // ← was NONE; NONE breaks tracking
        ip.depth_minimum_distance = 0.3f;   // suppress near-field SDK depth noise
        ip.coordinate_units   = UNIT::METER;
        ip.coordinate_system  = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;

        if (zed_.open(ip) != ERROR_CODE::SUCCESS) {
            RCLCPP_ERROR(get_logger(), "ZED open failed"); return;
        }

        // ── Positional tracking ──────────────────────────────────────────────
        PositionalTrackingParameters tp;
        tp.enable_area_memory = true;   // loop-closure for long sessions
        if (zed_.enablePositionalTracking(tp) != ERROR_CODE::SUCCESS) {
            RCLCPP_ERROR(get_logger(), "Positional tracking failed");
            zed_.close(); return;
        }

        lg_.create(720, 1280, CV_8U);
        rg_.create(720, 1280, CV_8U);

        RCLCPP_INFO(get_logger(),
            "ZED Hybrid SLAM UNDERWATER node started:\n"
            "  • DEPTH_MODE::PERFORMANCE (required for positional tracking init;\n"
            "    SDK depth output is never retrieved — SGBM replaces it)\n"
            "  • Positional tracking ON (loop-closure, area memory)\n"
            "  • SGBM z_max=5 m, voxel=4 cm, map_max=300k voxels\n"
            "  • Publishing: zed/point_cloud (PointCloud2)  zed/slam_mesh (Marker)");

        timer_ = create_wall_timer(std::chrono::milliseconds(33),
            std::bind(&ZedSlamNode::tick, this));
        init_ok = true;
    }

    ~ZedSlamNode() {
        zed_.disablePositionalTracking();
        zed_.close();
        cv::destroyAllWindows();
    }

private:

    // ──────────────────────────────────────────────────────────────────────────
    void tick()
    {
        if (zed_.grab() != ERROR_CODE::SUCCESS) return;
        ++frame_;

        // ── Camera pose (SDK visual odometry) ───────────────────────────────
        Pose pose;
        auto track_state = zed_.getPosition(pose, REFERENCE_FRAME::WORLD);

        if (frame_ % 30 == 0) {
            auto t = pose.getTranslation();
            RCLCPP_INFO(get_logger(),
                "[Frame %4d] tracking=%-12s | pos=(%.3f, %.3f, %.3f) m | "
                "map_voxels=%zu",
                frame_, toString(track_state).c_str(),
                t.x, t.y, t.z, map_.size());
        }

        // ── Retrieve raw stereo frames ────────────────────────────────────────
        Mat lz, rz;
        zed_.retrieveImage(lz, VIEW::LEFT);
        zed_.retrieveImage(rz, VIEW::RIGHT);
        cv::Mat left (lz.getHeight(), lz.getWidth(), CV_8UC4,
                      lz.getPtr<sl::uchar1>(sl::MEM::CPU));
        cv::Mat right(rz.getHeight(), rz.getWidth(), CV_8UC4,
                      rz.getPtr<sl::uchar1>(sl::MEM::CPU));

        cv::cvtColor(left,  lg_, cv::COLOR_BGRA2GRAY);
        cv::cvtColor(right, rg_, cv::COLOR_BGRA2GRAY);
        cv::Mat lb_bgr, rb_bgr;
        cv::cvtColor(left,  lb_bgr, cv::COLOR_BGRA2BGR);
        cv::cvtColor(right, rb_bgr, cv::COLOR_BGRA2BGR);

        // ── Rectify ──────────────────────────────────────────────────────────
        cv::Mat lr_g, rr_g, lrb, rrb;
        cv::remap(lg_,     lr_g, m1x_,m1y_, cv::INTER_LINEAR);
        cv::remap(rg_,     rr_g, m2x_,m2y_, cv::INTER_LINEAR);
        cv::remap(lb_bgr,  lrb,  m1x_,m1y_, cv::INTER_LINEAR);
        cv::remap(rb_bgr,  rrb,  m2x_,m2y_, cv::INTER_LINEAR);

        // ── Underwater preprocessing ──────────────────────────────────────────
        cv::Mat lrb_prep, rrb_prep;
        underwaterPreprocess(lrb,  lrb_prep, clahe_);
        underwaterPreprocess(rrb,  rrb_prep, clahe_);

        cv::Mat lr_green, rr_green;
        cv::extractChannel(lrb_prep, lr_green, 1);  // green = best UW channel
        cv::extractChannel(rrb_prep, rr_green, 1);

        // ── SGBM disparity ───────────────────────────────────────────────────
        cv::Mat dl, dr, df;
        lm_->compute(lr_green, rr_green, dl);
        rm_->compute(rr_green, lr_green, dr);
        wls_->filter(dl, lr_g, df, dr);
        df.convertTo(dfl_, CV_32F, 1.0/16.0);
        cv::threshold(dfl_, dfl_, 0.f, 0.f, cv::THRESH_TOZERO);

        // Border mask — SGBM unreliable within 80 px of border
        {
            const int BRD = 80;
            if (dfl_.rows > 2*BRD && dfl_.cols > 2*BRD) {
                dfl_(cv::Rect(0,           0,   BRD,      dfl_.rows)).setTo(0.f);
                dfl_(cv::Rect(dfl_.cols-BRD,0,  BRD,      dfl_.rows)).setTo(0.f);
                dfl_(cv::Rect(0,           0,   dfl_.cols,BRD       )).setTo(0.f);
                dfl_(cv::Rect(0,dfl_.rows-BRD,  dfl_.cols,BRD       )).setTo(0.f);
            }
        }

        cv::Mat dfl_med;
        cv::medianBlur(dfl_, dfl_med, 3);

        // ── Disparity temporal EMA (α=0.3, same as zed_rviz.cpp) ─────────────
        const float ALPHA = 0.3f;
        if (dfl_smooth_.empty()) {
            dfl_med.copyTo(dfl_smooth_);
        } else {
            for (int r = 0; r < dfl_smooth_.rows; ++r) {
                float* sm  = dfl_smooth_.ptr<float>(r);
                const float* cur = dfl_med.ptr<float>(r);
                for (int c = 0; c < dfl_smooth_.cols; ++c) {
                    if (cur[c] > 0.f)
                        sm[c] = (sm[c] > 0.f)
                                ? ALPHA*cur[c] + (1.f-ALPHA)*sm[c]
                                : cur[c];
                }
            }
        }

        // ── Reproject disparity → 3-D camera-space points ────────────────────
        cv::reprojectImageTo3D(dfl_smooth_, pts3D_, Q_, true, CV_32F);

        // ── Accumulate valid points into world-space voxel map ────────────────
        //   Only when tracking is OK (not UNAVAILABLE — that means lost/degraded)
        bool tracking_ok = (track_state == POSITIONAL_TRACKING_STATE::OK);

        if (tracking_ok) {
            const sl::Transform& tf = pose.pose_data;
            const int rows = pts3D_.rows, cols = pts3D_.cols;

            // Sample every MAP_STEP-th pixel in both axes
            for (int r = MAP_STEP/2; r < rows; r += MAP_STEP) {
                const cv::Vec3f* prow = pts3D_.ptr<cv::Vec3f>(r);
                const cv::Vec3b* crow = lrb_prep.ptr<cv::Vec3b>(r);
                for (int c = MAP_STEP/2; c < cols; c += MAP_STEP) {
                    const cv::Vec3f& p = prow[c];
                    float px = p[0], py = p[1], pz = p[2];
                    if (!std::isfinite(pz) || pz < 0.2f || pz > 5.0f) continue;
                    if (!std::isfinite(px) || !std::isfinite(py))       continue;

                    // Transform SGBM camera-space → ZED world-space
                    float wx, wy, wz;
                    sgbmToWorld(px, py, pz, tf, wx, wy, wz);

                    const cv::Vec3b& bgr = crow[c];
                    float r_f = bgr[2]/255.f;
                    float g_f = bgr[1]/255.f;
                    float b_f = bgr[0]/255.f;

                    if (!map_.insert(wx, wy, wz, r_f, g_f, b_f)) break; // map full
                }
            }
        }

        // ── Publish + render map every MAP_PUB_EVERY frames ─────────────────
        if (frame_ % MAP_PUB_EVERY == 0 && map_.size() > 0) {
            // Build flat vector
            std::vector<float> flat;
            map_.toVector(flat);

            // Push to GL thread
            {
                std::lock_guard<std::mutex> lk(g_pts.mtx);
                g_pts.pts = flat;
                g_pts.fresh.store(true);
            }

            publishPC(flat);
            publishMarker(flat);

            if (frame_ % 90 == 0) {
                RCLCPP_INFO(get_logger(),
                    "Map published — %zu voxels (%.1f k pts)",
                    map_.size(), map_.size() / 1000.0);
            }
        }

        // ── Previews ─────────────────────────────────────────────────────────
        // CLAHE enhanced colour — every frame
        cv::Mat small;
        cv::resize(lrb_prep, small, cv::Size(640, 360));
        cv::imshow("CLAHE colour [UW SLAM]", small);

        // Disparity — every 10 frames (throttled to avoid flicker)
        if (frame_ % 10 == 0) {
            double mn, mx;
            cv::minMaxLoc(dfl_, &mn, &mx, nullptr, nullptr, dfl_ > 0);
            if (mx > mn) {
                cv::Mat d8;
                dfl_.convertTo(d8, CV_8U, 255.0/(mx-mn), -255.0*mn/(mx-mn));
                cv::Mat dc; cv::applyColorMap(d8, dc, cv::COLORMAP_TURBO);
                cv::Mat dcs; cv::resize(dc, dcs, cv::Size(640, 360));
                cv::imshow("Disparity (SGBM/WLS) [UW SLAM]", dcs);
            }
        }

        cv::waitKey(1);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Publish accumulated world-space map as PointCloud2 (ROS REP-103 frame)
    void publishPC(const std::vector<float>& flat)
    {
        if (flat.empty()) return;
        const size_t N = flat.size() / 6;

        sensor_msgs::msg::PointCloud2 msg;
        msg.header.stamp    = get_clock()->now();
        msg.header.frame_id = "map";   // world frame — not camera frame
        msg.height          = 1;
        msg.width           = (uint32_t)N;
        msg.is_dense        = true;

        sensor_msgs::PointCloud2Modifier mod(msg);
        mod.setPointCloud2Fields(4,
            "x",   1, sensor_msgs::msg::PointField::FLOAT32,
            "y",   1, sensor_msgs::msg::PointField::FLOAT32,
            "z",   1, sensor_msgs::msg::PointField::FLOAT32,
            "rgb", 1, sensor_msgs::msg::PointField::FLOAT32);
        mod.resize(N);

        sensor_msgs::PointCloud2Iterator<float>   ix(msg,"x"), iy(msg,"y"), iz(msg,"z");
        sensor_msgs::PointCloud2Iterator<uint8_t> ir(msg,"rgb");

        for (size_t i = 0; i < flat.size(); i += 6) {
            float wx = flat[i], wy = flat[i+1], wz = flat[i+2];
            float rx, ry, rz;
            toROS(wx, wy, wz, rx, ry, rz);
            *ix = rx; *iy = ry; *iz = rz;
            ir[0] = (uint8_t)(flat[i+3]*255.f);  // R
            ir[1] = (uint8_t)(flat[i+4]*255.f);  // G
            ir[2] = (uint8_t)(flat[i+5]*255.f);  // B
            ++ix; ++iy; ++iz; ++ir;
        }
        pc_pub_->publish(msg);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Publish as Marker POINTS (for backward compat with existing RViz configs)
    void publishMarker(const std::vector<float>& flat)
    {
        if (flat.empty()) return;
        const size_t N = flat.size() / 6;

        visualization_msgs::msg::Marker m;
        m.header.stamp    = get_clock()->now();
        m.header.frame_id = "map";
        m.ns  = "slam_mesh";
        m.id  = 0;
        m.type   = visualization_msgs::msg::Marker::POINTS;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.scale.x = m.scale.y = VoxelMap::CELL;  // point size = voxel size
        m.color.a = 1.f;

        m.points.resize(N);
        m.colors.resize(N);

        for (size_t i = 0, v = 0; v < N; ++v, i += 6) {
            float wx = flat[i], wy = flat[i+1], wz = flat[i+2];
            float rx, ry, rz;
            toROS(wx, wy, wz, rx, ry, rz);
            m.points[v].x = rx; m.points[v].y = ry; m.points[v].z = rz;
            std_msgs::msg::ColorRGBA c;
            c.r = flat[i+3]; c.g = flat[i+4]; c.b = flat[i+5]; c.a = 1.f;
            m.colors[v] = c;
        }
        marker_pub_->publish(m);
    }

    // ── Members ───────────────────────────────────────────────────────────────
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    pc_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr  marker_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    sl::Camera zed_;

    // Underwater calibration
    cv::Mat Q_, m1x_, m1y_, m2x_, m2y_;

    // SGBM stereo pipeline
    cv::Ptr<cv::StereoSGBM>                   lm_;
    cv::Ptr<cv::StereoMatcher>                 rm_;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter>  wls_;
    cv::Ptr<cv::CLAHE>                         clahe_;

    cv::Mat lg_, rg_;          // raw gray buffers
    cv::Mat dfl_;              // current disparity (float)
    cv::Mat dfl_smooth_;       // temporally smoothed disparity
    cv::Mat pts3D_;            // 3-D points in camera frame

    // World-space voxel map
    VoxelMap map_;

    int frame_ = 0;

    // Pixel stride for voxel insertion (every MAP_STEP-th pixel per axis)
    static constexpr int MAP_STEP     = 6;   // 1/36 of pixels → fast enough
    // Publish / GL update every this many frames
    static constexpr int MAP_PUB_EVERY = 45;
};

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    if (!glfwInit()) { std::cerr << "GLFW init failed\n"; return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    GLFWwindow* win = glfwCreateWindow(1280, 720,
                        "ZED Hybrid SLAM — Underwater Spatial Map", nullptr, nullptr);
    if (!win) { std::cerr << "GLFW window failed\n"; glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    glfwSetScrollCallback(win,      scroll_cb);
    glfwSetMouseButtonCallback(win, mouse_btn_cb);
    glfwSetCursorPosCallback(win,   cursor_cb);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.02f, 0.05f, 0.12f, 1.f);   // deep blue underwater background

    auto node = std::make_shared<ZedSlamNode>();
    if (!node->init_ok) {
        rclcpp::shutdown();
        glfwDestroyWindow(win);
        glfwTerminate();
        return 1;
    }
    std::thread ros_thread([&]{ rclcpp::spin(node); });

    std::vector<float> render_pts;
    while (!glfwWindowShouldClose(win))
    {
        if (g_pts.fresh.load()) {
            std::lock_guard<std::mutex> lk(g_pts.mtx);
            render_pts = g_pts.pts;
            g_pts.fresh.store(false);
        }

        int W, H; glfwGetFramebufferSize(win, &W, &H);
        glViewport(0, 0, W, H);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Projection
        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        float asp = (H > 0) ? (float)W / H : 1.f;
        float t   = 0.1f * std::tan(60.f * 3.14159f / 360.f);
        glFrustum(-t*asp, t*asp, -t, t, 0.1f, 100.f);

        // Camera orbit view matrix
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        float yr = g_cam.yaw   * 3.14159f / 180.f;
        float pr = g_cam.pitch * 3.14159f / 180.f;
        float ecx = g_cam.cx + g_cam.dist*std::cos(pr)*std::sin(yr);
        float ecy = g_cam.cy + g_cam.dist*std::sin(pr);
        float ecz = g_cam.cz + g_cam.dist*std::cos(pr)*std::cos(yr);
        float fx=g_cam.cx-ecx, fy=g_cam.cy-ecy, fz=g_cam.cz-ecz;
        float fl=std::sqrt(fx*fx+fy*fy+fz*fz);
        fx/=fl; fy/=fl; fz/=fl;
        float sx=-fz, sy=0, sz=fx;
        float sl2=std::sqrt(sx*sx+sz*sz); sx/=sl2; sz/=sl2;
        float ux=sy*fz-sz*fy, uy=sz*fx-sx*fz, uz=sx*fy-sy*fx;
        float M[16]={
            sx,ux,-fx,0, sy,uy,-fy,0, sz,uz,-fz,0,
            -(sx*ecx+sy*ecy+sz*ecz),
            -(ux*ecx+uy*ecy+uz*ecz),
             (fx*ecx+fy*ecy+fz*ecz), 1
        };
        glLoadMatrixf(M);

        // Coordinate axes (XYZ = RGB)
        glLineWidth(1.5f);
        glBegin(GL_LINES);
          glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(0.5f,0,0);
          glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,0.5f,0);
          glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,0.5f);
        glEnd();

        // Accumulated world-space point cloud
        drawPoints(render_pts);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    rclcpp::shutdown();
    ros_thread.join();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
