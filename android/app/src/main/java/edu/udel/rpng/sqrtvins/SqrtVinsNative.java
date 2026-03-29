package edu.udel.rpng.sqrtvins;

public class SqrtVinsNative {
    static {
        System.loadLibrary("sqrt_vins_native");
    }

    /**
     * Initializes the VIO manager.
     * @param configPath Path to the configuration file.
     * @return True if initialization was successful.
     */
    public native boolean init(String configPath);

    /**
     * Feeds IMU data into the VIO manager.
     * @param timestamp Timestamp in seconds.
     * @param gx Gyroscope X reading (rad/s).
     * @param gy Gyroscope Y reading (rad/s).
     * @param gz Gyroscope Z reading (rad/s).
     * @param ax Accelerometer X reading (m/s^2).
     * @param ay Accelerometer Y reading (m/s^2).
     * @param az Accelerometer Z reading (m/s^2).
     */
    public native void feedImu(double timestamp, double gx, double gy, double gz, double ax, double ay, double az);

    /**
     * Feeds camera image into the VIO manager.
     * @param timestamp Timestamp in seconds.
     * @param matAddr Address of the OpenCV Mat object (cv::Mat).
     */
    public native void feedImage(double timestamp, long matAddr);

    /**
     * Retrieves the current pose (qx, qy, qz, qw, tx, ty, tz).
     * @return Pose as a double array, or null if not initialized.
     */
    public native double[] getPose();
}
