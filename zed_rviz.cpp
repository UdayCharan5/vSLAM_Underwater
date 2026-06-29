/**
 * Custom Underwater Mesh SLAM 
 * Combines ZED Positional Tracking with Custom OpenCV SGBM Meshing.
 * Uses Centroid Spatial Hashing to build a persistent, global 3D Triangle Map.
 */

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <yaml-cpp/yaml.h>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GLFW/glfw3.h>

#include <rclcpp/rclcpp.hpp>
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

// ─── Coordinate & Vector Math ─────────────────────────────────────────────────
// ZED RIGHT_HANDED_Y_UP (metres): X=right, Y=up, Z=backward
// ROS REP-103           (metres): X=forward, Y=left, Z=up
inline void toROS(float px, float py, float pz, float& X, float& Y, float& Z) {
    X = -pz; Y = -px; Z =  py;
}

inline sl::Translation rotateVector(const sl::Orientation& q, float vx, float vy, float vz) {
    float qx = q.ox, qy = q.oy, qz = q.oz, qw = q.ow;
    float tx = 2.0f * (qy * vz - qz * vy);
    float ty = 2.0f * (qz * vx - qx * vz);
    float tz = 2.0f * (qx * vy - qy * vx);
    float rx = vx + qw * tx + (qy * tz - qz * ty);
    float ry = vy + qw * ty + (qz * tx - qx * tz);
    float rz = vz + qw * tz + (qx * ty - qy * tx);
    return sl::Translation(rx, ry, rz);
}

inline sl::Translation transformPoint(const sl::Translation& local_pt, const sl::Translation& cam_t, const sl::Orientation& cam_q) {
    sl::Translation rotated = rotateVector(cam_q, local_pt.x, local_pt.y, local_pt.z);
    return sl::Translation(cam_t.x + rotated.x, cam_t.y + rotated.y, cam_t.z + rotated.z);
}

// ─── Camera Orbit State (GLFW) ────────────────────────────────────────────────
struct Camera3D {
    float yaw=0.f, pitch=-25.f, dist=5.f;
    float cx=0, cy=0, cz=0;
    double last_mx=-1, last_my=-1;
    bool dragging=false;
} g_cam;

void scroll_cb(GLFWwindow*,double,double dy) { g_cam.dist-=(float)dy*0.3f; g_cam.dist=std::max(0.5f,g_cam.dist); }
void mouse_btn_cb(GLFWwindow*,int btn,int action,int) { if(btn==GLFW_MOUSE_BUTTON_LEFT) g_cam.dragging=(action==GLFW_PRESS); }
void cursor_cb(GLFWwindow*,double x,double y){
    if(g_cam.dragging&&g_cam.last_mx>=0){
        g_cam.yaw  +=(float)(x-g_cam.last_mx)*0.4f;
        g_cam.pitch+=(float)(y-g_cam.last_my)*0.4f;
        g_cam.pitch=std::max(-89.f,std::min(89.f,g_cam.pitch));
    }
    g_cam.last_mx=x; g_cam.last_my=y;
}

// ─── Shared Mesh (ROS → GL) ───────────────────────────────────────────────────
struct MeshData {
    std::vector<float> tris;   // GL preview (ZED native Y-UP frame)
    std::vector<float> ros;    // ROS publish (REP-103 frame)
    std::atomic<bool>  fresh{false};
    std::mutex         mtx;
} g_mesh;

// ─── Global Face Spatial Hashing ──────────────────────────────────────────────
// Stores exactly one triangle per 3cm voxel to prevent overlapping faces
struct GlobalTri {
    cv::Vec3f v[3]; // Vertices
    cv::Vec3f c[3]; // Colors
};

struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& o) const { return x==o.x && y==o.y && z==o.z; }
};

struct VoxelKeyHash {
    size_t operator()(const VoxelKey& k) const {
        return (size_t)k.x * 73856093ULL ^ (size_t)k.y * 19349663ULL ^ (size_t)k.z * 83492791ULL;
    }
};

// ─── YAML & Image Preprocessing ───────────────────────────────────────────────
static cv::Mat yamlToMat(const YAML::Node& node, int rows, int cols, int type) {
    cv::Mat m(rows, cols, type);
    int k = 0;
    for (const auto& v : node) m.at<double>(k / cols, k % cols) = v.as<double>(), ++k;
    return m;
}

static void underwaterPreprocess(const cv::Mat& bgr_in, cv::Mat& bgr_out, cv::Ptr<cv::CLAHE>& clahe) {
    std::vector<cv::Mat> ch(3);
    cv::split(bgr_in, ch);
    for(int i = 0; i < 3; ++i){
        cv::Mat f32; ch[i].convertTo(f32, CV_32F);
        std::vector<float> v(f32.begin<float>(), f32.end<float>());
        std::nth_element(v.begin(), v.begin() + int(0.01f*v.size()), v.end());
        float lo = v[int(0.01f*v.size())];
        std::nth_element(v.begin(), v.begin() + int(0.99f*v.size()), v.end());
        float hi = v[int(0.99f*v.size())];
        if(hi <= lo) continue;
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

// ─── ROS2 Node ────────────────────────────────────────────────────────────────
class ZedSlamMapNode : public rclcpp::Node {
public:
    ZedSlamMapNode() : Node("zed_slam_map") {
        marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("zed/slam_mesh", 10);

        const std::string YAML_PATH = "/home/uday/ros2_ws/src/zed_rviz/scripts/underwater_calib.yaml";
        if(!std::filesystem::exists(YAML_PATH)){
            RCLCPP_FATAL(get_logger(), "Underwater calib missing: %s", YAML_PATH.c_str());
            rclcpp::shutdown(); return;
        }

        YAML::Node uw = YAML::LoadFile(YAML_PATH)["underwater"];
        cv::Mat K1 = yamlToMat(uw["left"]["K"], 3, 3, CV_64F);
        cv::Mat K2 = yamlToMat(uw["right"]["K"], 3, 3, CV_64F);
        cv::Mat D1 = yamlToMat(uw["left"]["D"], 1, 8, CV_64F).colRange(0,5);
        cv::Mat D2 = yamlToMat(uw["right"]["D"], 1, 8, CV_64F).colRange(0,5);
        cv::Mat R1 = yamlToMat(uw["R1"], 3, 3, CV_64F);
        cv::Mat R2 = yamlToMat(uw["R2"], 3, 3, CV_64F);
        cv::Mat P1 = yamlToMat(uw["P1"], 3, 4, CV_64F);
        cv::Mat P2 = yamlToMat(uw["P2"], 3, 4, CV_64F);
        Q_         = yamlToMat(uw["Q"],  4, 4, CV_64F);

        const cv::Size IMG_SZ(1280, 720);
        cv::initUndistortRectifyMap(K1,D1,R1,P1, IMG_SZ, CV_32FC1, m1x_,m1y_);
        cv::initUndistortRectifyMap(K2,D2,R2,P2, IMG_SZ, CV_32FC1, m2x_,m2y_);

        InitParameters ip;
        ip.camera_resolution = RESOLUTION::HD720;
        ip.camera_fps        = 30;
        ip.depth_mode        = DEPTH_MODE::PERFORMANCE; // REQUIRED for positional tracking
        ip.coordinate_units  = UNIT::METER;
        ip.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; 

        if(zed_.open(ip) != ERROR_CODE::SUCCESS){
            RCLCPP_ERROR(get_logger(),"ZED open failed"); rclcpp::shutdown(); return;
        }

        PositionalTrackingParameters tp;
        tp.enable_area_memory = true;
        if (zed_.enablePositionalTracking(tp) != ERROR_CODE::SUCCESS) {
            RCLCPP_ERROR(get_logger(), "Positional tracking failed"); zed_.close(); rclcpp::shutdown(); return;
        }

        int bs = 5, nd = 256;
        lm_ = cv::StereoSGBM::create(0, nd, bs, 8*bs*bs, 32*bs*bs, 1, 0, 8, 60, 2, cv::StereoSGBM::MODE_HH);
        rm_  = cv::ximgproc::createRightMatcher(lm_);
        wls_ = cv::ximgproc::createDisparityWLSFilter(lm_);
        wls_->setLambda(8000); wls_->setSigmaColor(1.5);
        clahe_ = cv::createCLAHE(2.5, cv::Size(8, 8));

        lg_.create(720,1280,CV_8U); rg_.create(720,1280,CV_8U);

        timer_=create_wall_timer(std::chrono::milliseconds(33), std::bind(&ZedSlamMapNode::tick,this));
        RCLCPP_INFO(get_logger(), "Underwater Custom Mesh SLAM Started");
    }

    ~ZedSlamMapNode() {
        zed_.disablePositionalTracking();
        zed_.close();
    }

private:
    void tick() {
        if(zed_.grab() != ERROR_CODE::SUCCESS) return;
        frame_++;

        sl::Pose pose;
        POSITIONAL_TRACKING_STATE state = zed_.getPosition(pose, REFERENCE_FRAME::WORLD);
        if (state != POSITIONAL_TRACKING_STATE::OK) return; // Wait for tracking to stabilize

        sl::Translation cam_t = pose.getTranslation();
        sl::Orientation cam_q = pose.getOrientation();

        Mat lz,rz;
        zed_.retrieveImage(lz,VIEW::LEFT); zed_.retrieveImage(rz,VIEW::RIGHT);
        cv::Mat left(lz.getHeight(),lz.getWidth(),CV_8UC4,lz.getPtr<sl::uchar1>(sl::MEM::CPU));
        cv::Mat right(rz.getHeight(),rz.getWidth(),CV_8UC4,rz.getPtr<sl::uchar1>(sl::MEM::CPU));

        cv::cvtColor(left, lg_,cv::COLOR_BGRA2GRAY); cv::cvtColor(right,rg_,cv::COLOR_BGRA2GRAY);
        cv::cvtColor(left, lb_,cv::COLOR_BGRA2BGR);

        cv::remap(lg_,lr_, m1x_,m1y_,cv::INTER_LINEAR);
        cv::remap(rg_,rr_, m2x_,m2y_,cv::INTER_LINEAR);
        cv::remap(lb_,lrb_, m1x_,m1y_,cv::INTER_LINEAR);

        cv::Mat rb_bgr, rrb;
        cv::cvtColor(right, rb_bgr, cv::COLOR_BGRA2BGR);
        cv::remap(rb_bgr, rrb, m2x_, m2y_, cv::INTER_LINEAR);

        cv::Mat lrb_prep, rrb_prep;
        underwaterPreprocess(lrb_, lrb_prep, clahe_);
        underwaterPreprocess(rrb, rrb_prep, clahe_);

        cv::Mat lr_g, rr_g;
        cv::extractChannel(lrb_prep, lr_g, 1); cv::extractChannel(rrb_prep, rr_g, 1);

        cv::Mat dl,dr,df;
        lm_->compute(lr_g, rr_g, dl); rm_->compute(rr_g, lr_g, dr);
        wls_->filter(dl, lr_, df, dr);
        df.convertTo(dfl_,CV_32F,1.0/16.0);
        cv::threshold(dfl_, dfl_, 0.f, 0.f, cv::THRESH_TOZERO);

        const int BRD_LEFT = 256, BRD_OTHER = 80; 
        if(dfl_.rows > 2*BRD_OTHER && dfl_.cols > BRD_LEFT + BRD_OTHER) {
            dfl_(cv::Rect(0, 0, BRD_LEFT, dfl_.rows)).setTo(0.f);
            dfl_(cv::Rect(dfl_.cols-BRD_OTHER, 0, BRD_OTHER, dfl_.rows)).setTo(0.f);
            dfl_(cv::Rect(0, 0, dfl_.cols, BRD_OTHER)).setTo(0.f);
            dfl_(cv::Rect(0, dfl_.rows-BRD_OTHER, dfl_.cols, BRD_OTHER)).setTo(0.f);
        }

        cv::Mat dfl_med; cv::medianBlur(dfl_,dfl_med,3);

        const float DISP_ALPHA = 0.3f;
        if(dfl_smooth_.empty()) { dfl_med.copyTo(dfl_smooth_); } 
        else {
            for(int r=0;r<dfl_smooth_.rows;++r){
                float* sm  = dfl_smooth_.ptr<float>(r);
                const float* cur = dfl_med.ptr<float>(r);
                for(int c=0;c<dfl_smooth_.cols;++c){
                    if(cur[c] > 0.f) sm[c] = (sm[c] > 0.f) ? DISP_ALPHA*cur[c] + (1.f-DISP_ALPHA)*sm[c] : cur[c];
                }
            }
        }

        cv::reprojectImageTo3D(dfl_smooth_, pts3D_raw_, Q_, true, CV_32F);

        // Build Local Mesh
        const int STEP = 8;
        cv::Mat pd_raw, cd_raw;
        cv::resize(pts3D_raw_, pd_raw, cv::Size(), 1.0/STEP, 1.0/STEP, cv::INTER_NEAREST);
        cv::resize(lrb_prep, cd_raw, cv::Size(), 1.0/STEP, 1.0/STEP, cv::INTER_LINEAR);

        std::vector<float> local_tris;
        buildLocalMesh(pd_raw, cd_raw, local_tris);

        // Accumulate into Global Mesh
        integrateGlobalMesh(local_tris, cam_t, cam_q);

        // Update Render/Publish Buffers every 10 frames
        if (frame_ % 10 == 0) {
            updateMeshBuffers();
        }
    }

    void buildLocalMesh(const cv::Mat& pts3D, const cv::Mat& col, std::vector<float>& out) {
        const int rows=pts3D.rows, cols_=pts3D.cols;
        std::vector<int> vidx(rows*cols_,-1);
        std::vector<float> vx,vy,vz,vr,vg,vb;

        for(int r=0;r<rows;++r){
            const cv::Vec3f* prow=pts3D.ptr<cv::Vec3f>(r);
            const cv::Vec3b* crow=col.ptr<cv::Vec3b>(r);
            for(int c=0;c<cols_;++c){
                float X_cv=prow[c][0], Y_cv=prow[c][1], Z_cv=prow[c][2];
                if(!std::isfinite(Z_cv)||Z_cv<0.4f||Z_cv>5.0f) continue;
                
                vidx[r*cols_+c]=(int)vx.size();
                vx.push_back(X_cv); vy.push_back(Y_cv); vz.push_back(Z_cv);
                vr.push_back(crow[c][2]/255.f); vg.push_back(crow[c][1]/255.f); vb.push_back(crow[c][0]/255.f);
            }
        }

        auto eOk=[&](int a,int b)->bool{
            if(a<0||b<0) return false;
            float dx=vx[a]-vx[b], dy=vy[a]-vy[b], dz=vz[a]-vz[b];
            float thresh = 0.5f*(vz[a]+vz[b]) * 0.07f;
            return (dx*dx+dy*dy+dz*dz) < thresh*thresh;
        };

        auto push3=[&](int a,int b,int c2){
            out.push_back(vx[a]); out.push_back(vy[a]); out.push_back(vz[a]);
            out.push_back(vr[a]); out.push_back(vg[a]); out.push_back(vb[a]);
            out.push_back(vx[b]); out.push_back(vy[b]); out.push_back(vz[b]);
            out.push_back(vr[b]); out.push_back(vg[b]); out.push_back(vb[b]);
            out.push_back(vx[c2]); out.push_back(vy[c2]); out.push_back(vz[c2]);
            out.push_back(vr[c2]); out.push_back(vg[c2]); out.push_back(vb[c2]);
        };

        for(int r=0;r<rows-1;++r) for(int c=0;c<cols_-1;++c){
            int tl=vidx[r*cols_+c], tr=vidx[r*cols_+c+1];
            int bl=vidx[(r+1)*cols_+c], br=vidx[(r+1)*cols_+c+1];
            if(tl<0&&tr<0) continue;
            if(bl<0&&br<0) continue;
            if(tl>=0&&tr>=0&&bl>=0&&eOk(tl,tr)&&eOk(tr,bl)&&eOk(bl,tl)) push3(tl,tr,bl);
            if(tr>=0&&br>=0&&bl>=0&&eOk(tr,br)&&eOk(br,bl)&&eOk(bl,tr)) push3(tr,br,bl);
        }
    }

    void integrateGlobalMesh(const std::vector<float>& local_tris, const sl::Translation& cam_t, const sl::Orientation& cam_q) {
        for (size_t i = 0; i < local_tris.size(); i += 18) {
            GlobalTri tri;
            sl::Translation centroid(0,0,0);

            for (int v = 0; v < 3; ++v) {
                size_t b = i + v * 6;
                // Convert OpenCV (X right, Y down, Z forward) to ZED Y-UP (X right, Y up, Z backward)
                sl::Translation p_local(local_tris[b], -local_tris[b+1], -local_tris[b+2]);
                
                // Transform to World Frame
                sl::Translation p_world = transformPoint(p_local, cam_t, cam_q);
                
                tri.v[v] = cv::Vec3f(p_world.x, p_world.y, p_world.z);
                tri.c[v] = cv::Vec3f(local_tris[b+3], local_tris[b+4], local_tris[b+5]); // RGB
                
                centroid.x += p_world.x; centroid.y += p_world.y; centroid.z += p_world.z;
            }

            centroid.x /= 3.0f; centroid.y /= 3.0f; centroid.z /= 3.0f;

            // 3cm Voxel Hash based on Face Centroid
            float VOXEL_SIZE = 0.03f;
            VoxelKey key{(int)std::floor(centroid.x / VOXEL_SIZE), (int)std::floor(centroid.y / VOXEL_SIZE), (int)std::floor(centroid.z / VOXEL_SIZE)};

            // Deduplicate: Only add if voxel is empty
            if (global_faces_.find(key) == global_faces_.end()) {
                global_faces_[key] = tri;
            }
        }
    }

    void updateMeshBuffers() {
        std::vector<float> gl_out, ros_out;
        gl_out.reserve(global_faces_.size() * 18);
        ros_out.reserve(global_faces_.size() * 18);

        for (const auto& kv : global_faces_) {
            const GlobalTri& tri = kv.second;
            for (int v = 0; v < 3; ++v) {
                // GL Buffer (Y-UP)
                gl_out.push_back(tri.v[v][0]); gl_out.push_back(tri.v[v][1]); gl_out.push_back(tri.v[v][2]);
                gl_out.push_back(tri.c[v][0]); gl_out.push_back(tri.c[v][1]); gl_out.push_back(tri.c[v][2]);

                // ROS Buffer (REP-103)
                float rx, ry, rz;
                toROS(tri.v[v][0], tri.v[v][1], tri.v[v][2], rx, ry, rz);
                ros_out.push_back(rx); ros_out.push_back(ry); ros_out.push_back(rz);
                ros_out.push_back(tri.c[v][0]); ros_out.push_back(tri.c[v][1]); ros_out.push_back(tri.c[v][2]);
            }
        }

        {
            std::lock_guard<std::mutex> lk(g_mesh.mtx);
            g_mesh.tris = std::move(gl_out);
            g_mesh.ros  = std::move(ros_out);
            g_mesh.fresh.store(true);
        }

        publishMarker();
    }

    void publishMarker() {
        std::vector<float> snap;
        {
            std::lock_guard<std::mutex> lk(g_mesh.mtx);
            snap = g_mesh.ros;
        }
        if (snap.empty()) return;

        visualization_msgs::msg::Marker m;
        m.header.stamp    = get_clock()->now();
        m.header.frame_id = "odom";
        m.ns     = "slam_mesh";
        m.id     = 0;
        m.type   = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.scale.x = m.scale.y = m.scale.z = 1.0;
        m.color.a = 0.85f;

        size_t numTri = snap.size() / 18;
        m.points.resize(numTri * 3);
        m.colors.resize(numTri * 3);

        size_t out = 0;
        for (size_t i = 0; i < snap.size(); i += 18) {
            for (int v = 0; v < 3; ++v) {
                size_t b = i + v*6;
                geometry_msgs::msg::Point p;
                p.x=snap[b]; p.y=snap[b+1]; p.z=snap[b+2];
                m.points[out] = p;
                
                std_msgs::msg::ColorRGBA c;
                c.r=snap[b+3]; c.g=snap[b+4]; c.b=snap[b+5]; c.a=0.85f;
                m.colors[out] = c;
                ++out;
            }
        }
        marker_pub_->publish(m);
    }

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    sl::Camera zed_;
    int frame_ = 0;

    std::unordered_map<VoxelKey, GlobalTri, VoxelKeyHash> global_faces_;

    cv::Ptr<cv::StereoSGBM>  lm_;
    cv::Ptr<cv::StereoMatcher> rm_;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_;
    cv::Mat Q_, m1x_, m1y_, m2x_, m2y_;
    cv::Ptr<cv::CLAHE> clahe_;

    cv::Mat lg_, rg_, lb_, lr_, rr_, lrb_, dfl_, dfl_smooth_, pts3D_raw_;
};

// ─── OpenGL Draw ──────────────────────────────────────────────────────────────
void drawMesh(const std::vector<float>& tris) {
    if (tris.empty()) return;

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glLineWidth(0.8f);
    glBegin(GL_TRIANGLES);
    for (size_t i = 0; i < tris.size(); i += 6) {
        glColor3f(tris[i+3]*0.6f+0.3f, tris[i+4]*0.6f+0.3f, tris[i+5]*0.6f+0.3f);
        glVertex3f(tris[i], tris[i+1], tris[i+2]);
    }
    glEnd();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_TRIANGLES);
    for (size_t i = 0; i < tris.size(); i += 6) {
        glColor4f(tris[i+3], tris[i+4], tris[i+5], 0.45f);
        glVertex3f(tris[i], tris[i+1], tris[i+2]);
    }
    glEnd();
    glDisable(GL_BLEND);
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    if (!glfwInit()) { std::cerr << "GLFW init failed\n"; return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    GLFWwindow* win = glfwCreateWindow(1280, 720, "Underwater Custom Mesh SLAM", nullptr, nullptr);
    if (!win) { std::cerr << "GLFW window failed\n"; glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    glfwSetScrollCallback(win,     scroll_cb);
    glfwSetMouseButtonCallback(win, mouse_btn_cb);
    glfwSetCursorPosCallback(win,   cursor_cb);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.05f, 0.05f, 0.10f, 1.f);

    auto node = std::make_shared<ZedSlamMapNode>();
    std::thread ros_thread([&]{ rclcpp::spin(node); });

    std::vector<float> render_tris;
    while (!glfwWindowShouldClose(win)) {
        if (g_mesh.fresh.load()) {
            std::lock_guard<std::mutex> lk(g_mesh.mtx);
            render_tris = g_mesh.tris;
            g_mesh.fresh.store(false);
        }

        int W, H; glfwGetFramebufferSize(win, &W, &H);
        glViewport(0, 0, W, H);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        float asp = (H > 0) ? (float)W / H : 1.f;
        float t   = 0.1f * std::tan(60.f * 3.14159f / 360.f);
        glFrustum(-t*asp, t*asp, -t, t, 0.1f, 200.f);

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

        glLineWidth(2.0f);
        glBegin(GL_LINES);
          glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(0.5f,0,0);
          glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,0.5f,0);
          glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,0.5f);
        glEnd();

        drawMesh(render_tris);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    rclcpp::shutdown();
    ros_thread.join();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
