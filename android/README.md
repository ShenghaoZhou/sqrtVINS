# sqrt-VINS for Android

This directory contains the necessary structure and JNI bridge to build and run **sqrt-VINS** on Android.

## Getting Started

To use sqrt-VINS in your Android application, you'll need the following:

1.  **Android Studio** with **NDK** and **CMake** installed.
2.  **OpenCV for Android** (prebuilt version is recommended).
3.  **Eigen** (header-only, can be downloaded and provided as a path).

## Build Instructions

1.  Open the `android/` directory in Android Studio.
2.  Update `android/app/src/main/cpp/CMakeLists.txt` with the correct paths for **OpenCV** and **Eigen**.
3.  Implement the sensor feed (IMU and Camera) in your Java/Kotlin code, calling the methods in `SqrtVinsNative`.

### Sensor Feed Example (Java)

```java
SqrtVinsNative vins = new SqrtVinsNative();

// 1. Initialize
vins.init(configPath);

// 2. Feed IMU data (from SensorManager)
vins.feedImu(timestamp, gx, gy, gz, ax, ay, az);

// 3. Feed Camera image (e.g., from Camera2 API)
vins.feedImage(timestamp, matAddr);

// 4. Get current pose
double[] pose = vins.getPose();
if (pose != null) {
    // Current translation: (pose[4], pose[5], pose[6])
    // Current orientation: (pose[0], pose[1], pose[2], pose[3])
}
```

## Performance Tips

sqrt-VINS is highly optimized for embedded systems and runs efficiently even on ARM-based mobile devices.

-   **Floating-point optimization**: The filter is designed to work with 32-bit single precision, which is significantly faster on mobile CPUs. Ensure `-DUSE_FLOAT=1` is set in CMake (this is the default).
-   **Hardware acceleration**: OpenCV for Android can leverage NEON instructions and GPU (via OpenCL/Vulkan) for image processing if available.
-   **Threading**: IMU and Camera feeds should ideally be handled on separate threads, or consistently synchronized.
