#include <ros/ros.h>
#include<std_msgs/Float64.h>
#include<nav_msgs/Odometry.h>
#include<math.h>
#include<tf/transform_datatypes.h>
#include<shell_simulation/Car_Command.h>
#include<vector>

using namespace std;

// Structure for coordinates
struct coordinates
{
     float x,y;
};

// Class for goal/target points
struct points{
    float x, y;
    int direction;
};

// Store all target points in a vector
vector < points > list_of_target(12);
void set_targets(float x, float y, float dir, int count)
{
     list_of_target[count].x = x;
     list_of_target[count].y = y;
     list_of_target[count].direction = dir;
}

// Global variables
coordinates position, destination;
double roll, pitch, yaw;
int axis, run = 0, point_count = 0; // X+ = 1, X- = -1, Y+ = 2, Y- = -2
float speed, speed_x, speed_y;
const float pi = 3.14159;


// Get next coordinates
void get_destination()
{
     point_count++;
     if(point_count == 12)
          {
               while(ros::ok())
               {
               }
          }
     destination.x = list_of_target[point_count].x;
     destination.y = list_of_target[point_count].y;
     axis = list_of_target[point_count].direction;
}

// Odom topic callback
void get_position(const nav_msgs::Odometry& msg)
{
     position.x = msg.pose.pose.position.x;
     position.y = msg.pose.pose.position.y;
     speed_x = msg.twist.twist.linear.x;
     speed_y = msg.twist.twist.linear.y;
     speed = sqrt(speed_x*speed_x + speed_y*speed_y);

     tf::Quaternion quat_form(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
          msg.pose.pose.orientation.z, msg.pose.pose.orientation.w);

     tf::Matrix3x3 quat_matrix(quat_form);
     quat_matrix.getRPY(roll, pitch, yaw);

     // Convert yaw so that it is measured from X axis (rather than Y which is default)
     if(yaw < (pi/2))
          yaw = (pi/2) - yaw;
     else
          yaw = (5*pi/2) - yaw;
}

int main( int argc, char **argv)
{
     // Set goal points
     set_targets(-212,0,-1, 0);
     set_targets(-212,-128,-2, 1);
     set_targets(44,-128,1, 2);
     set_targets(44,-256,-2, 3);
     set_targets(-212,-256,-1, 4);
     set_targets(-212,-48,2, 5);
     set_targets(-84,-48,1, 6);
     set_targets(-84,0,2, 7);
     set_targets(44,0,1, 8);
     set_targets(44,-256,-2, 9);
     set_targets(-84,-256,-1, 10);
     set_targets(-84,-180,2, 11);

     // Set values for first point
     destination.x = list_of_target[0].x, destination.y = list_of_target[0].y, axis = list_of_target[0].direction;

     // Initialize node
     ros::init (argc, argv, "shell_simulation_node");
     ros::NodeHandle nh;

     // Create subscriber and Publisher objects for communication with each different topic
     // ros::Subscriber sub_motion = nh.subscribe("motion", 100, &get_destination);
     ros::Subscriber sub_odom = nh.subscribe("odom", 100, &get_position);
     ros::Publisher pub = nh.advertise<shell_simulation::Car_Command>("airsim_node/PhysXCar/car_cmd_body_frame", 100);

     // Initialize variables
     float distance;
     coordinates del;
     float cross_track_error, heading_error;
     int heading_sense, cross_track_sense, flag, flag2;
     shell_simulation::Car_Command vel;
     float prop_heading = 0.55, prop_cros_track = 0.45, speed_calib, heading_calib, cross_track_calib;
     float target_speed = 15;

     ros::Duration(2).sleep();

     // ros::Rate rate(0.02);

     // Start loop to keep the node running
     while(ros::ok())
     {
          // Check and execute callbacks from any of the subscribed topics
          ros::spinOnce();

          // Record the coordinates followed by vehicle
          // ROS_INFO_STREAM_THROTTLE(2, "x: " << position.x << " y: " << position.y);

          // Set default values
          vel.handbrake = false;
          vel.brake = 0;
          vel.steering = 0;
          vel.throttle = 0;

          // Calculating distance from destination
          del.x = destination.x - position.x;
          del.y = destination.y - position.y;

          // Calculating cross_track and heading errors , direction/signs considered
          if (abs(axis) == 1)
          {
               // If X axis
               distance = abs(del.x);
               flag = (yaw > pi)*(axis == 1);
               cross_track_error = (destination.y - position.y)*(axis);
               heading_error = pi*(2*flag + (-axis+1)/2) - yaw;
          }
          else
          {
               // If Y axis
               distance = abs(del.y);
               cross_track_error = (position.x - destination.x)*(axis/2);
               flag = (yaw > 3*pi/2)*(axis == 2);
               flag2 = (yaw < pi/2)*(axis == -2);
               heading_error = (pi/2)*(-0.5*axis + 2) - ((flag == 0) - (flag > 0))*yaw + 2*pi*(flag - flag2);
          }

          // Calculate the angular sense of errors about Z axis
          heading_sense = (heading_error > 0) - (heading_error < 0);
          cross_track_sense = (cross_track_error > 0) - (cross_track_error < 0);

          // Calculation of linear velocity component
          if(distance > 14)
          {
               heading_calib = (-2.925)*abs(heading_error)*abs(heading_error) + (-0.0875)*abs(heading_error) + 1;
               cross_track_calib = -abs(cross_track_error)*(0.3) + 1;
               vel.throttle = 0.6*heading_calib*cross_track_calib;

               if(vel.throttle < 0.3)
                    vel.throttle = 0.35;
          }
          else
          {

               vel.throttle = 0;
               vel.steering = 0;
               vel.brake = 0.3;

               // Brake until speed is reduced to 0.7
               while(speed > 5.25)
               {
                    ros::spinOnce();
                    pub.publish(vel);
               }

               vel.brake = 0;
               pub.publish(vel);
               get_destination();
               continue;
          }

          // Calculation of angular velocity component
          if(abs(heading_error) > (pi/8.5))
          {
               target_speed = 10;
               if(abs(cross_track_error) > 10)
               {
                    vel.steering = 0;
                    vel.throttle = 0.3;
               }
               else
               {
                    vel.steering = 0.85*heading_sense*(abs(cross_track_error)*(-0.08) + 1);
                    // vel.throttle = 0.8*(target_speed - speed)*(target_speed > speed)/target_speed;
                    vel.throttle = 0.52;
               }
          }
          else
          {
               vel.steering = prop_heading*(heading_error/2) + prop_cros_track*(cross_track_error/9.0);
          }

          if(abs(vel.steering > 1))
               vel.steering = 0.9*heading_sense;

          // As positive steering means right -> negative angular, thus sign has to be interchanged
          vel.steering = -vel.steering;

          // Publish calculated velocity values
          pub.publish(vel);
          // rate.sleep();
     }
}
