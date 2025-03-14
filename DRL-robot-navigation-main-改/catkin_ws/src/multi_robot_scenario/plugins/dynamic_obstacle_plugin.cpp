#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>

namespace gazebo {
  class DynamicObstaclePlugin : public ModelPlugin {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) {
      // 存储模型指针
      this->model = _parent;
      
      // 获取模型名称来确定运动类型
      std::string modelName = this->model->GetName();
      
      // 初始化参数
      this->period = 0;
      this->initialPos = this->model->WorldPose();
      
      // 根据模型名称设置不同的运动模式
      if (modelName == "circular_obstacle") {
        this->motionType = "circular";
        this->center = ignition::math::Vector3d(2, 2, 0.5);
        this->radius = 1.5;
        this->speed = 0.5;
      }
      else if (modelName == "linear_obstacle") {
        this->motionType = "linear";
        this->startPoint = ignition::math::Vector3d(-3, -2, 0.5);
        this->endPoint = ignition::math::Vector3d(-3, 2, 0.5);
        this->speed = 0.4;
        this->forward = true;
      }
      else if (modelName == "rectangular_obstacle") {
        this->motionType = "rectangular";
        this->waypoints = {
          ignition::math::Vector3d(0, 3, 0.5),
          ignition::math::Vector3d(2, 3, 0.5),
          ignition::math::Vector3d(2, 1, 0.5),
          ignition::math::Vector3d(0, 1, 0.5)
        };
        this->currentWaypoint = 0;
        this->speed = 0.3;
      }
      else if (modelName == "eight_shaped_obstacle") {
        this->motionType = "eight";
        this->center = ignition::math::Vector3d(3, -2, 0.5);
        this->radius = 1.0;
        this->speed = 0.6;
      }

      // 连接更新事件
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&DynamicObstaclePlugin::OnUpdate, this));
    }

    public: void OnUpdate() {
      this->period += 0.001; // 假设更新频率为1000Hz

      if (this->motionType == "circular") {
        // 圆周运动
        double angle = this->speed * this->period;
        ignition::math::Vector3d pos(
          this->center.X() + this->radius * cos(angle),
          this->center.Y() + this->radius * sin(angle),
          this->center.Z()
        );
        this->model->SetWorldPose(
          ignition::math::Pose3d(pos, this->initialPos.Rot())
        );
      }
      else if (this->motionType == "linear") {
        // 直线往复运动
        ignition::math::Vector3d currentPos = this->model->WorldPose().Pos();
        if (this->forward) {
          if (currentPos.Y() >= this->endPoint.Y())
            this->forward = false;
          else
            currentPos += ignition::math::Vector3d(0, this->speed * 0.001, 0);
        } else {
          if (currentPos.Y() <= this->startPoint.Y())
            this->forward = true;
          else
            currentPos += ignition::math::Vector3d(0, -this->speed * 0.001, 0);
        }
        this->model->SetWorldPose(
          ignition::math::Pose3d(currentPos, this->initialPos.Rot())
        );
      }
      else if (this->motionType == "rectangular") {
        // 矩形轨迹运动
        ignition::math::Vector3d currentPos = this->model->WorldPose().Pos();
        ignition::math::Vector3d targetPos = this->waypoints[this->currentWaypoint];
        ignition::math::Vector3d diff = targetPos - currentPos;
        
        if (diff.Length() < 0.1) {
          this->currentWaypoint = (this->currentWaypoint + 1) % this->waypoints.size();
        } else {
          diff.Normalize();
          currentPos += diff * this->speed * 0.001;
          this->model->SetWorldPose(
            ignition::math::Pose3d(currentPos, this->initialPos.Rot())
          );
        }
      }
      else if (this->motionType == "eight") {
        // "8"字运动
        double t = this->speed * this->period;
        ignition::math::Vector3d pos(
          this->center.X() + this->radius * sin(t),
          this->center.Y() + this->radius * sin(2 * t) / 2,
          this->center.Z()
        );
        this->model->SetWorldPose(
          ignition::math::Pose3d(pos, this->initialPos.Rot())
        );
      }
    }

    private: physics::ModelPtr model;
    private: event::ConnectionPtr updateConnection;
    private: std::string motionType;
    private: ignition::math::Pose3d initialPos;
    private: double period;
    private: double speed;
    private: double radius;
    private: bool forward;
    private: ignition::math::Vector3d center;
    private: ignition::math::Vector3d startPoint;
    private: ignition::math::Vector3d endPoint;
    private: std::vector<ignition::math::Vector3d> waypoints;
    private: int currentWaypoint;
  };

  // 注册插件
  GZ_REGISTER_MODEL_PLUGIN(DynamicObstaclePlugin)
}
