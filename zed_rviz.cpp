//  zed_rviz.cpp  —  Live stereo → SGBM+WLS depth → structured 3D mesh
//                   *** UNDERWATER VERSION ***
//
//  Changes from the aerial version:
//    • Calibration loaded from scripts/underwater_calib.yaml
//      (not from the ZED SDK at runtime).  Underwater focal length is ~25 %
//      larger than in air due to the flat-port refractive index shift.
//    • Pre-computed R1, R2, P1, P2, Q from the YAML → no stereoRectify call.
//    • SGBM tuned for underwater: higher uniqueness ratio, stronger WLS
//      regularisation, reduced max disparity for cleaner low-contrast matches.
//    • Simplified real-time Sea-thru colour correction (Akkaynak & Treibitz,
//      2019) using the per-pixel depth map to recover attenuated red/green.
//    • z_max reduced from 8 m → 5 m (typical underwater visibility limit).
//    • Adaptive edge threshold widened slightly (15 % vs 12 %) to bridge the
//      wider disparity gaps caused by low underwater texture.
//
//  Coordinate convention (toWorld):
//    X = -px     (stereo baseline flipped, already in metres)
//    Y = -py     (image rows down → world Y up, already in metres)
//    Z =  pz     (depth in metres — underwater Q built with metre-scale object pts)
//
//  NOTE: The aerial version divided by 1000 here because the ZED SDK Q matrix
//  used mm units.  The underwater Q was calibrated with SQUARE_SIZE_M in metres,
//  so reprojectImageTo3D already returns metres — NO extra ×1e-3 needed.

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
#include <fstream>
#include <cmath>
#include <iostream>
#include <string>
#include <filesystem>

using namespace sl;

// ─── Sea-thru attenuation coefficients (β per channel, 1/metre) ───────────────

// ─── Coordinate convention ────────────────────────────────────────────────────
// Underwater Q (from stereoRectify with metre-scale object points) means
// reprojectImageTo3D already returns XYZ in metres — do NOT divide by 1000.
inline void toWorld(float px, float py, float pz,
                    float& X, float& Y, float& Z)
{
    X = -px;   // metres, flip X for camera→world baseline direction
    Y = -py;   // metres, flip Y so rows-down becomes Y-up
    Z =  pz;   // metres, depth unchanged
}

// ─── YAML helper — load a flat list into a cv::Mat ───────────────────────────
static cv::Mat yamlToMat(const YAML::Node& node, int rows, int cols, int type)
{
    cv::Mat m(rows, cols, type);
    int k = 0;
    for (const auto& v : node)
        m.at<double>(k / cols, k % cols) = v.as<double>(), ++k;
    return m;
}

// ─── Camera orbit ─────────────────────────────────────────────────────────────
struct Camera3D {
    float yaw=0.f, pitch=-25.f, dist=5.f;
    float cx=0, cy=0, cz=2;
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

// ─── Shared mesh (ROS → GL) ───────────────────────────────────────────────────
struct MeshData {
    std::vector<float> tris;
    std::atomic<bool>  fresh{false};
    std::mutex         mtx;
} g_mesh;

// ─── buildMesh ────────────────────────────────────────────────────────────────
// Edge threshold 10 % of Z — tighter than 15 % to prevent spikes at depth
// discontinuities.  Underwater scenes have lower contrast but wider depth
// jumps between objects, so a tight threshold avoids bridging foreground→
// background across real edges.
void buildMesh(const cv::Mat& pts3D, const cv::Mat& col,
               std::vector<float>& out,
               float z_min=0.3f, float z_max=5.f)
{
    const int rows=pts3D.rows, cols_=pts3D.cols;

    std::vector<int>   vidx(rows*cols_,-1);
    std::vector<float> vx,vy,vz,vr,vg,vb;
    vx.reserve(rows*cols_); vy.reserve(rows*cols_); vz.reserve(rows*cols_);
    vr.reserve(rows*cols_); vg.reserve(rows*cols_); vb.reserve(rows*cols_);

    for(int r=0;r<rows;++r){
        const cv::Vec3f* prow=pts3D.ptr<cv::Vec3f>(r);
        const cv::Vec3b* crow=col.ptr<cv::Vec3b>(r);
        for(int c=0;c<cols_;++c){
            const cv::Vec3f& p=prow[c];
            float X=p[0],Y=p[1],Z=p[2];
            if(!std::isfinite(Z)||Z<z_min||Z>z_max) continue;
            if(!std::isfinite(X)||!std::isfinite(Y)) continue;
            float maxR=Z*1.2f;
            if(X*X+Y*Y>=maxR*maxR) continue;
            vidx[r*cols_+c]=(int)vx.size();
            const cv::Vec3b& bgr=crow[c];
            vx.push_back(X); vy.push_back(Y); vz.push_back(Z);
            vr.push_back(bgr[2]/255.f);
            vg.push_back(bgr[1]/255.f);
            vb.push_back(bgr[0]/255.f);
        }
    }

    // Adaptive edge: 7 % of average Z.
    // Tight threshold prevents triangles from bridging real depth discontinuities
    // (foreground object edge → background wall) which is the main source of spikes.
    auto eOk=[&](int a,int b)->bool{
        if(a<0||b<0) return false;
        float dx=vx[a]-vx[b],dy=vy[a]-vy[b],dz=vz[a]-vz[b];
        float d2=dx*dx+dy*dy+dz*dz;
        float avgZ=0.5f*(vz[a]+vz[b]);
        float thresh=avgZ*0.07f;   // 7 % — tight to kill spikes at depth edges
        return d2 < thresh*thresh;
    };

    auto push3=[&](int a,int b,int c2){
        out.push_back(vx[a]);out.push_back(vy[a]);out.push_back(vz[a]);
        out.push_back(vr[a]);out.push_back(vg[a]);out.push_back(vb[a]);
        out.push_back(vx[b]);out.push_back(vy[b]);out.push_back(vz[b]);
        out.push_back(vr[b]);out.push_back(vg[b]);out.push_back(vb[b]);
        out.push_back(vx[c2]);out.push_back(vy[c2]);out.push_back(vz[c2]);
        out.push_back(vr[c2]);out.push_back(vg[c2]);out.push_back(vb[c2]);
    };

    out.clear();
    out.reserve((rows-1)*(cols_-1)*2*18);

    for(int r=0;r<rows-1;++r) for(int c=0;c<cols_-1;++c){
        int tl=vidx[ r   *cols_+c  ];
        int tr=vidx[ r   *cols_+c+1];
        int bl=vidx[(r+1)*cols_+c  ];
        int br=vidx[(r+1)*cols_+c+1];

        // Require at least 3 valid corners to attempt any triangle.
        // Each triangle is checked independently with eOk so depth-discontinuity
        // edges are still rejected.  The valid==3 hole-fill (which caused spikes)
        // is NOT included — each triangle only uses its own 3 corners.
        if(tl<0&&tr<0) continue;
        if(bl<0&&br<0) continue;

        // Upper triangle: tl–tr–bl
        if(tl>=0&&tr>=0&&bl>=0&&eOk(tl,tr)&&eOk(tr,bl)&&eOk(bl,tl))
            push3(tl,tr,bl);
        // Lower triangle: tr–br–bl
        if(tr>=0&&br>=0&&bl>=0&&eOk(tr,br)&&eOk(br,bl)&&eOk(bl,tr))
            push3(tr,br,bl);
    }
}

// ─── OpenGL draw ──────────────────────────────────────────────────────────────
void drawMesh(const std::vector<float>& tris)
{
    if(tris.empty()) return;
    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    glLineWidth(0.8f);
    glBegin(GL_TRIANGLES);
    for(size_t i=0;i<tris.size();i+=6){
        glColor3f(tris[i+3]*0.6f+0.3f,tris[i+4]*0.6f+0.3f,tris[i+5]*0.6f+0.3f);
        glVertex3f(tris[i],tris[i+1],tris[i+2]);
    }
    glEnd();
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_TRIANGLES);
    for(size_t i=0;i<tris.size();i+=6){
        glColor4f(tris[i+3],tris[i+4],tris[i+5],0.55f);
        glVertex3f(tris[i],tris[i+1],tris[i+2]);
    }
    glEnd();
    glDisable(GL_BLEND);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
}

// ─── CLAHE colour enhancement for display ─────────────────────────────────────
// Apply CLAHE in the CIE L*a*b* colour space: only the L (lightness) channel
// is equalised, so hues are preserved while contrast and brightness increase.
// This makes the blueish/greenish underwater image appear sharper and more
// natural without the artefacts of Sea-thru's per-pixel inversion.
static void claheEnhanceColor(const cv::Mat& bgr_in, cv::Mat& bgr_out,
                               cv::Ptr<cv::CLAHE>& clahe)
{
    cv::Mat lab;
    cv::cvtColor(bgr_in, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> ch(3);
    cv::split(lab, ch);
    clahe->apply(ch[0], ch[0]);   // enhance L channel only
    cv::merge(ch, lab);
    cv::cvtColor(lab, bgr_out, cv::COLOR_Lab2BGR);
}


// ─── Underwater image preprocessing (matches Python Underwater_depth_live.py) ────
// Per-channel percentile stretch (1st–99th) + CLAHE + unsharp mask.
// This is the same pipeline as underwater_preprocess() in the Python reference:
//   1. Stretch each channel to fill 0–255 using 1st/99th percentile clipping
//      (adapts to any ambient light level, removes global colour cast)
//   2. Apply CLAHE per channel (local contrast enhancement)
//   3. Unsharp mask (sharpens edges, helps SGBM texture detection)
// For SGBM matching we extract only the GREEN channel from the result
// (green is the least optically attenuated colour underwater).
static void underwaterPreprocess(const cv::Mat& bgr_in, cv::Mat& bgr_out,
                                  cv::Ptr<cv::CLAHE>& clahe)
{
    std::vector<cv::Mat> ch(3);
    cv::split(bgr_in, ch);
    for(int i = 0; i < 3; ++i){
        cv::Mat f32; ch[i].convertTo(f32, CV_32F);
        // 1st/99th percentile
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
    // Unsharp mask (matches Python: addWeighted(out,1.4, blur,-0.4, 0))
    cv::Mat blur;
    cv::GaussianBlur(bgr_out, blur, cv::Size(0,0), 1.5);
    cv::addWeighted(bgr_out, 1.4, blur, -0.4, 0.0, bgr_out);
}


// ─── ROS2 node ────────────────────────────────────────────────────────────────
class ZedNode : public rclcpp::Node {
public:
    ZedNode() : Node("zed_pointcloud_publisher") {
        publisher_  = create_publisher<sensor_msgs::msg::PointCloud2>("zed/point_cloud",10);
        marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("zed/mesh_marker",10);

        // ── Find this executable's directory to locate the YAML ───────────────
        // The YAML is always at  <package>/scripts/underwater_calib.yaml.
        // We resolve it relative to a known path so it works both from the
        // build tree and installed.
        const std::string YAML_PATH =
            "/home/uday/ros2_ws/src/zed_rviz/scripts/underwater_calib.yaml";

        if(!std::filesystem::exists(YAML_PATH)){
            RCLCPP_FATAL(get_logger(),
                "Underwater calibration file not found: %s", YAML_PATH.c_str());
            rclcpp::shutdown(); return;
        }

        // ── Load underwater_calib.yaml ────────────────────────────────────────
        YAML::Node cfg;
        try { cfg = YAML::LoadFile(YAML_PATH); }
        catch(const std::exception& e){
            RCLCPP_FATAL(get_logger(),"YAML parse error: %s", e.what());
            rclcpp::shutdown(); return;
        }

        YAML::Node uw = cfg["underwater"];

        // ── Intrinsics ────────────────────────────────────────────────────────
        // K is stored as [fx, 0, cx, 0, fy, cy, 0, 0, 1] (row-major 3×3)
        cv::Mat K1 = yamlToMat(uw["left"]["K"],  3, 3, CV_64F);
        cv::Mat K2 = yamlToMat(uw["right"]["K"], 3, 3, CV_64F);

        // D stored as 8-element vector [k1,k2,p1,p2,k3,0,0,0]
        cv::Mat D1_8 = yamlToMat(uw["left"]["D"],  1, 8, CV_64F);
        cv::Mat D2_8 = yamlToMat(uw["right"]["D"], 1, 8, CV_64F);
        // cv::initUndistortRectifyMap only uses the first 5 coefficients
        cv::Mat D1 = D1_8.colRange(0,5);
        cv::Mat D2 = D2_8.colRange(0,5);

        // ── Pre-computed rectification outputs (no stereoRectify call needed) ─
        cv::Mat R1 = yamlToMat(uw["R1"], 3, 3, CV_64F);
        cv::Mat R2 = yamlToMat(uw["R2"], 3, 3, CV_64F);
        cv::Mat P1 = yamlToMat(uw["P1"], 3, 4, CV_64F);
        cv::Mat P2 = yamlToMat(uw["P2"], 3, 4, CV_64F);
        Q_         = yamlToMat(uw["Q"],  4, 4, CV_64F);

        RCLCPP_INFO(get_logger(),
            "Underwater calib loaded: fx_L=%.2f  fx_R=%.2f  baseline=%.4f m",
            K1.at<double>(0,0), K2.at<double>(0,0),
            uw["baseline_m"].as<double>());

        // depth_scale_factor: with Option A (underwater Q), scale = 1.0.
        // We read it anyway and log it for reference.
        depth_scale_ = cfg["depth_scale_factor"] ?
                       cfg["depth_scale_factor"].as<float>() : 1.0f;
        RCLCPP_INFO(get_logger(),
            "depth_scale_factor=%.4f  (applied=false — underwater Q already correct)",
            depth_scale_);

        // ── Build rectification maps from the underwater calibration ──────────
        const cv::Size IMG_SZ(1280, 720);
        cv::initUndistortRectifyMap(K1,D1,R1,P1, IMG_SZ, CV_32FC1, m1x_,m1y_);
        cv::initUndistortRectifyMap(K2,D2,R2,P2, IMG_SZ, CV_32FC1, m2x_,m2y_);

        // ── Open ZED (images only — depth from SGBM) ─────────────────────────
        InitParameters ip;
        ip.camera_resolution = RESOLUTION::HD720;
        ip.camera_fps        = 30;
        ip.depth_mode        = DEPTH_MODE::NONE;
        if(zed_.open(ip)!=ERROR_CODE::SUCCESS){
            RCLCPP_ERROR(get_logger(),"ZED open failed"); rclcpp::shutdown(); return;
        }

        // ── SGBM parameters (aligned with Python Underwater_depth_live.py reference) ──
        // Key changes vs previous:
        //   MODE_HH     — full 5-direction DP, better border/edge quality than 3WAY
        //   blockSize=5 — finer detail capture (Python uses 3 at ¼-res = ~12 full-res)
        //   P1/P2 scaled to blockSize (keeps same ratio as Python)
        //   uniquenessRatio=8 (matches Python default)
        //   speckleWindowSize=60 (Python default; 200 was over-smoothing valid clusters)
        int bs=5, nd=128;
        lm_=cv::StereoSGBM::create(0,nd,bs,
            8*bs*bs, 32*bs*bs,
            1, 0, 8,    // uniquenessRatio=8
            60, 2,      // speckleWindowSize=60 (Python default)
            cv::StereoSGBM::MODE_HH);  // full DP — better border/side quality
        rm_  = cv::ximgproc::createRightMatcher(lm_);
        wls_ = cv::ximgproc::createDisparityWLSFilter(lm_);
        wls_->setLambda(8000);      // Python uses 8000 (was 16000 — less over-smooth)
        wls_->setSigmaColor(1.5);

        // CLAHE for preprocessing (clipLimit=2.5 matches Python's _clahe setting)
        clahe_ = cv::createCLAHE(2.5, cv::Size(8, 8));

        lg_.create(720,1280,CV_8U);
        rg_.create(720,1280,CV_8U);

        timer_=create_wall_timer(std::chrono::milliseconds(33),
            std::bind(&ZedNode::tick,this));
        RCLCPP_INFO(get_logger(),"ZED UNDERWATER node running  [CLAHE stereo + colour enhancement]");
    }
    ~ZedNode(){ zed_.close(); cv::destroyAllWindows(); }

private:
    // ──────────────────────────────────────────────────────────────────────────
    void tick()
    {
        if(zed_.grab()!=ERROR_CODE::SUCCESS) return;

        Mat lz,rz;
        zed_.retrieveImage(lz,VIEW::LEFT);
        zed_.retrieveImage(rz,VIEW::RIGHT);
        cv::Mat left (lz.getHeight(),lz.getWidth(),CV_8UC4,lz.getPtr<sl::uchar1>(sl::MEM::CPU));
        cv::Mat right(rz.getHeight(),rz.getWidth(),CV_8UC4,rz.getPtr<sl::uchar1>(sl::MEM::CPU));

        cv::cvtColor(left, lg_,cv::COLOR_BGRA2GRAY);
        cv::cvtColor(right,rg_,cv::COLOR_BGRA2GRAY);
        cv::cvtColor(left, lb_,cv::COLOR_BGRA2BGR);

        // Rectify using UNDERWATER maps (loaded from YAML)
        cv::remap(lg_,lr_,  m1x_,m1y_,cv::INTER_LINEAR);
        cv::remap(rg_,rr_,  m2x_,m2y_,cv::INTER_LINEAR);
        cv::remap(lb_,lrb_, m1x_,m1y_,cv::INTER_LINEAR);

        // ── Underwater preprocessing (Python reference pipeline) ────────────────
        // Per-channel percentile stretch + CLAHE + unsharp on both rectified
        // colour images.  Extract GREEN channel for SGBM matching.
        // (Green is least attenuated underwater → best stereo matching SNR.)
        cv::Mat rb_bgr;
        cv::cvtColor(right, rb_bgr, cv::COLOR_BGRA2BGR);
        cv::Mat rrb;
        cv::remap(rb_bgr, rrb, m2x_, m2y_, cv::INTER_LINEAR);

        cv::Mat lrb_prep, rrb_prep;
        underwaterPreprocess(lrb_,  lrb_prep, clahe_);
        underwaterPreprocess(rrb,   rrb_prep, clahe_);

        cv::Mat lr_g, rr_g;
        cv::extractChannel(lrb_prep, lr_g, 1);   // green = index 1 in BGR
        cv::extractChannel(rrb_prep, rr_g, 1);

        // ── Disparity (SGBM on green; WLS guided by original gray) ─────────────
        cv::Mat dl,dr,df;
        lm_->compute(lr_g, rr_g, dl);
        rm_->compute(rr_g, lr_g, dr);
        wls_->filter(dl, lr_, df, dr);  // guide = original gray (real edges)
        df.convertTo(dfl_,CV_32F,1.0/16.0);

        // Zero out SGBM invalid markers
        cv::threshold(dfl_, dfl_, 0.f, 0.f, cv::THRESH_TOZERO);

        // ── Border masking — PRIMARY FIX FOR SIDE DISTORTIONS ──────────────────
        // SGBM is unreliable within ~(numDisparities + blockSize/2) pixels of
        // the image borders.  Border pixels with bad disparity back-project to
        // huge XY offsets (large u-cx or v-cy × small depth), creating the
        // "wings" / spikes growing from the sides of the mesh.
        {
            const int BRD = 80;
            if(dfl_.rows > 2*BRD && dfl_.cols > 2*BRD){
                dfl_(cv::Rect(0,            0,   BRD,       dfl_.rows)).setTo(0.f);
                dfl_(cv::Rect(dfl_.cols-BRD,0,   BRD,       dfl_.rows)).setTo(0.f);
                dfl_(cv::Rect(0,            0,   dfl_.cols, BRD       )).setTo(0.f);
                dfl_(cv::Rect(0, dfl_.rows-BRD,  dfl_.cols, BRD       )).setTo(0.f);
            }
        }

        // Median 3×3 — kills salt-pepper noise
        cv::Mat dfl_med;
        cv::medianBlur(dfl_,dfl_med,3);

        // ── Disparity-domain temporal EMA (Python TemporalSmoother) ────────────
        // Smooth DISPARITY (not 3D points) — same technique as Python reference.
        // Only valid (>0) pixels update the accumulator; invalid pixels keep the
        // previous estimate (no decay) so depth persists across brief occlusions.
        const float DISP_ALPHA = 0.3f;
        if(dfl_smooth_.empty()){
            dfl_med.copyTo(dfl_smooth_);
        } else {
            for(int r=0;r<dfl_smooth_.rows;++r){
                float*       sm  = dfl_smooth_.ptr<float>(r);
                const float* cur = dfl_med.ptr<float>(r);
                for(int c=0;c<dfl_smooth_.cols;++c){
                    if(cur[c] > 0.f){
                        // Cold-start: history zero → jump directly (no warm-up lag)
                        sm[c] = (sm[c] > 0.f)
                                  ? DISP_ALPHA*cur[c] + (1.f-DISP_ALPHA)*sm[c]
                                  : cur[c];
                    }
                    // else: keep previous valid disparity
                }
            }
        }

        // ── Reproject smoothed disparity to 3D ─────────────────────────────────
        cv::reprojectImageTo3D(dfl_smooth_, pts3D_raw_, Q_, true, CV_32F);

        if(pts3D_m_.empty()) pts3D_m_.create(pts3D_raw_.size(), CV_32FC3);
        for(int r=0;r<pts3D_raw_.rows;++r){
            const cv::Vec3f* src = pts3D_raw_.ptr<cv::Vec3f>(r);
            cv::Vec3f*       dst = pts3D_m_.ptr<cv::Vec3f>(r);
            for(int c=0;c<pts3D_raw_.cols;++c){
                float X,Y,Z;
                toWorld(src[c][0],src[c][1],src[c][2],X,Y,Z);
                dst[c]={X,Y,Z};
            }
        }
        // Disparity EMA handles temporal stability — copy directly to smooth buf
        pts3D_m_.copyTo(pts3D_smooth_);

        // ── Colour for mesh/PC: use the preprocessed left image ────────────────
        cv::Mat lrb_enh;
        lrb_prep.copyTo(lrb_enh);

        // ── Downsample for mesh ───────────────────────────────────────────────
        const int STEP=8;
        cv::Mat pd_raw,cd_raw;
        cv::resize(pts3D_smooth_,pd_raw,cv::Size(),1.0/STEP,1.0/STEP,cv::INTER_NEAREST);
        cv::resize(lrb_enh,      cd_raw,cv::Size(),1.0/STEP,1.0/STEP,cv::INTER_LINEAR);

        // Downsample for publish/CSV (enhanced colour)
        cv::resize(pts3D_smooth_, pts3D_pub_, cv::Size(),1.0/PC_STEP,1.0/PC_STEP,cv::INTER_NEAREST);
        cv::resize(lrb_enh,       lrb_pub_,   cv::Size(),1.0/PC_STEP,1.0/PC_STEP,cv::INTER_LINEAR);

        // Bilateral filter on Z channel of downsampled mesh points
        {
            int h=pd_raw.rows,w=pd_raw.cols;
            cv::Mat Zplane(h,w,CV_32F);
            for(int r=0;r<h;++r){
                const cv::Vec3f* p=pd_raw.ptr<cv::Vec3f>(r);
                float* z=Zplane.ptr<float>(r);
                for(int c=0;c<w;++c) z[c]=p[c][2];
            }
            cv::Mat Zfilt;
            cv::bilateralFilter(Zplane,Zfilt,5,0.05f,3.f);
            pd_.create(h,w,CV_32FC3);
            for(int r=0;r<h;++r){
                const cv::Vec3f* p=pd_raw.ptr<cv::Vec3f>(r);
                const float*     z=Zfilt.ptr<float>(r);
                cv::Vec3f*       o=pd_.ptr<cv::Vec3f>(r);
                for(int c=0;c<w;++c)
                    o[c]={p[c][0],p[c][1],z[c]};
            }
            cd_raw.copyTo(cd_);
        }

        // ── Build mesh ────────────────────────────────────────────────────────
        {
            std::vector<float> tris;
            buildMesh(pd_,cd_,tris);
            std::lock_guard<std::mutex> lk(g_mesh.mtx);
            g_mesh.tris=std::move(tris);
            g_mesh.fresh.store(true);
        }

        publishPC();
        publishMarker();

        //if(frame_%10==0) saveCSV(frame_);
        //++frame_;

        // ── Previews ─────────────────────────────────────────────────────────
        cv::Mat d8; double mn,mx;
        cv::minMaxLoc(dfl_,&mn,&mx,nullptr,nullptr,dfl_>0);
        if(mx>mn){
            dfl_.convertTo(d8,CV_8U,255.0/(mx-mn),-255.0*mn/(mx-mn));
            cv::Mat dc; cv::applyColorMap(d8,dc,cv::COLORMAP_TURBO);
            cv::imshow("Rectified Left (underwater calib)", lr_);
            cv::imshow("Disparity (WLS)",                  d8);
            cv::imshow("Depth colour",                     dc);
            cv::imshow("CLAHE enhanced colour",            lrb_enh);
        }
        cv::waitKey(1);
    }

    // ──────────────────────────────────────────────────────────────────────────
    void publishPC()
    {
        sensor_msgs::msg::PointCloud2 msg;
        msg.header.stamp=get_clock()->now(); msg.header.frame_id="zed_left_camera";
        msg.height=pts3D_pub_.rows; msg.width=pts3D_pub_.cols; msg.is_dense=false;
        sensor_msgs::PointCloud2Modifier mod(msg);
        mod.setPointCloud2Fields(4,
            "x",1,sensor_msgs::msg::PointField::FLOAT32,
            "y",1,sensor_msgs::msg::PointField::FLOAT32,
            "z",1,sensor_msgs::msg::PointField::FLOAT32,
            "rgb",1,sensor_msgs::msg::PointField::FLOAT32);
        mod.resize(pts3D_pub_.rows*pts3D_pub_.cols);
        sensor_msgs::PointCloud2Iterator<float>   ix(msg,"x"),iy(msg,"y"),iz(msg,"z");
        sensor_msgs::PointCloud2Iterator<uint8_t> ir(msg,"rgb");
        for(int r=0;r<pts3D_pub_.rows;++r){
            const cv::Vec3f* prow=pts3D_pub_.ptr<cv::Vec3f>(r);
            const cv::Vec3b* crow=lrb_pub_.ptr<cv::Vec3b>(r);
            for(int c=0;c<pts3D_pub_.cols;++c){
                float X=prow[c][0],Y=prow[c][1],Z=prow[c][2];
                bool ok=std::isfinite(Z)&&Z>0.3f&&Z<5.f&&    // 5 m max (underwater)
                        std::isfinite(X)&&std::isfinite(Y)&&
                        X*X+Y*Y<(Z*1.2f)*(Z*1.2f);
                *ix=ok?X:NAN; *iy=ok?Y:NAN; *iz=ok?Z:NAN;
                const cv::Vec3b& bgr=crow[c];
                ir[0]=bgr[2]; ir[1]=bgr[1]; ir[2]=bgr[0];
                ++ix;++iy;++iz;++ir;
            }
        }
        publisher_->publish(msg);
    }

    // ──────────────────────────────────────────────────────────────────────────
    void publishMarker()
    {
        std::vector<float> snap;
        { std::lock_guard<std::mutex> lk(g_mesh.mtx); snap=g_mesh.tris; }
        if(snap.empty()) return;

        visualization_msgs::msg::Marker m;
        m.header.stamp=get_clock()->now(); m.header.frame_id="zed_left_camera";
        m.ns="mesh"; m.id=0;
        m.type=visualization_msgs::msg::Marker::LINE_LIST;
        m.action=visualization_msgs::msg::Marker::ADD;
        m.scale.x=0.003; m.color.a=1;

        size_t numTri=snap.size()/18;
        m.points.resize(numTri*6);
        m.colors.resize(numTri*6);
        size_t out=0;
        for(size_t i=0;i<snap.size();i+=18){
            auto mp=[&](size_t b)->geometry_msgs::msg::Point{
                geometry_msgs::msg::Point p;
                p.x=snap[b];p.y=snap[b+1];p.z=snap[b+2]; return p;};
            auto mc=[&](size_t b)->std_msgs::msg::ColorRGBA{
                std_msgs::msg::ColorRGBA c;
                c.r=snap[b+3];c.g=snap[b+4];c.b=snap[b+5];c.a=1; return c;};
            auto pA=mp(i),pB=mp(i+6),pC=mp(i+12);
            auto cA=mc(i),cB=mc(i+6),cC=mc(i+12);
            m.points[out]=pA;m.colors[out]=cA;++out;
            m.points[out]=pB;m.colors[out]=cB;++out;
            m.points[out]=pB;m.colors[out]=cB;++out;
            m.points[out]=pC;m.colors[out]=cC;++out;
            m.points[out]=pC;m.colors[out]=cC;++out;
            m.points[out]=pA;m.colors[out]=cA;++out;
        }
        marker_pub_->publish(m);
    }

    // ──────────────────────────────────────────────────────────────────────────
    void saveCSV(int frame)
    {
        std::string fn="pointcloud_frame_"+std::to_string(frame)+".csv";
        std::ofstream f(fn); f<<"X,Y,Z,R,G,B\n";
        for(int r=0;r<pts3D_pub_.rows;++r){
            const cv::Vec3f* prow=pts3D_pub_.ptr<cv::Vec3f>(r);
            const cv::Vec3b* crow=lrb_pub_.ptr<cv::Vec3b>(r);
            for(int c=0;c<pts3D_pub_.cols;++c){
                float X=prow[c][0],Y=prow[c][1],Z=prow[c][2];
                if(!std::isfinite(X)||!std::isfinite(Y)||!std::isfinite(Z)) continue;
                const cv::Vec3b& bgr=crow[c];
                f<<X<<','<<Y<<','<<Z<<','
                 <<(int)bgr[2]<<','<<(int)bgr[1]<<','<<(int)bgr[0]<<'\n';
            }
        }
        RCLCPP_INFO(get_logger(),"Saved %s",fn.c_str());
    }

    // ── Members ───────────────────────────────────────────────────────────────
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr   publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr  marker_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    Camera zed_;
    cv::Ptr<cv::StereoSGBM>  lm_;
    cv::Ptr<cv::StereoMatcher> rm_;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_;
    cv::Mat Q_,m1x_,m1y_,m2x_,m2y_;
    float depth_scale_ = 1.0f;
    cv::Ptr<cv::CLAHE> clahe_;

    cv::Mat lg_,rg_,lb_;
    cv::Mat lr_,rr_,lrb_;
    cv::Mat dfl_;
    cv::Mat dfl_smooth_;   // disparity EMA accumulator (from Python TemporalSmoother)
    cv::Mat pts3D_raw_;
    cv::Mat pts3D_m_;
    cv::Mat pts3D_smooth_;
    cv::Mat pd_,cd_;
    cv::Mat pts3D_pub_;
    cv::Mat lrb_pub_;

    int frame_=0;
    static constexpr int PC_STEP = 8;
};


// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc,char** argv)
{
    rclcpp::init(argc,argv);

    if(!glfwInit()){std::cerr<<"GLFW init failed\n";return 1;}
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,1);
    GLFWwindow* win=glfwCreateWindow(1280,720,"Live 3D Spatial Mesh [UNDERWATER]",nullptr,nullptr);
    if(!win){std::cerr<<"GLFW window failed\n";glfwTerminate();return 1;}
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    glfwSetScrollCallback(win,scroll_cb);
    glfwSetMouseButtonCallback(win,mouse_btn_cb);
    glfwSetCursorPosCallback(win,cursor_cb);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.02f,0.05f,0.12f,1.f);   // deep-blue background for underwater theme

    auto node=std::make_shared<ZedNode>();
    std::thread ros_thread([&]{rclcpp::spin(node);});

    std::vector<float> render_tris;
    while(!glfwWindowShouldClose(win))
    {
        if(g_mesh.fresh.load()){
            std::lock_guard<std::mutex> lk(g_mesh.mtx);
            render_tris=g_mesh.tris;
            g_mesh.fresh.store(false);
        }

        int W,H; glfwGetFramebufferSize(win,&W,&H);
        glViewport(0,0,W,H);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);glLoadIdentity();
        float asp=(H>0)?(float)W/H:1.f;
        float t=0.1f*std::tan(60.f*3.14159f/360.f);
        glFrustum(-t*asp,t*asp,-t,t,0.1f,50.f);

        glMatrixMode(GL_MODELVIEW);glLoadIdentity();
        float yr=g_cam.yaw*3.14159f/180.f, pr=g_cam.pitch*3.14159f/180.f;
        float ecx=g_cam.cx+g_cam.dist*std::cos(pr)*std::sin(yr);
        float ecy=g_cam.cy+g_cam.dist*std::sin(pr);
        float ecz=g_cam.cz+g_cam.dist*std::cos(pr)*std::cos(yr);
        float fx=g_cam.cx-ecx,fy=g_cam.cy-ecy,fz=g_cam.cz-ecz;
        float fl=std::sqrt(fx*fx+fy*fy+fz*fz);
        fx/=fl;fy/=fl;fz/=fl;
        float sx=-fz,sy=0,sz=fx;
        float sl=std::sqrt(sx*sx+sz*sz); sx/=sl;sz/=sl;
        float ux=sy*fz-sz*fy,uy=sz*fx-sx*fz,uz=sx*fy-sy*fx;
        float M[16]={sx,ux,-fx,0,sy,uy,-fy,0,sz,uz,-fz,0,
            -(sx*ecx+sy*ecy+sz*ecz),-(ux*ecx+uy*ecy+uz*ecz),fx*ecx+fy*ecy+fz*ecz,1};
        glLoadMatrixf(M);

        glLineWidth(1.5f);
        glBegin(GL_LINES);
          glColor3f(1,0,0);glVertex3f(0,0,0);glVertex3f(0.5f,0,0);
          glColor3f(0,1,0);glVertex3f(0,0,0);glVertex3f(0,0.5f,0);
          glColor3f(0,0,1);glVertex3f(0,0,0);glVertex3f(0,0,0.5f);
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
