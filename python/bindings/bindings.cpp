#include <cstdint>
#include <pybind11/attr.h>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>

#include <pybind11/eigen.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>

#include "pool_slam/HalTypes.hpp"
#include "pool_slam/System.hpp"
#include "pool_slam/SystemTypes.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pool_slam, m) {
    m.doc() = "Python bindings for PoolSLAM";

    // --- Sensor Ingestion Structs (HalTypes.hpp) ---
    py::class_<PoolSLAM::Imu>(m, "Imu")
        .def(py::init<>())
        .def(py::init([](double timestamp,
                         const decltype(PoolSLAM::Imu::a)& a,
                         const decltype(PoolSLAM::Imu::w)& w,
                         decltype(PoolSLAM::Imu::t) t) {
            PoolSLAM::Imu sample{};
            sample.timestamp = timestamp;
            sample.a = a;
            sample.w = w;
            sample.t = t;
            return sample;
        }), py::arg("timestamp"), py::arg("a"), py::arg("w"), py::arg("t") = 0.0f)
        .def_readwrite("timestamp", &PoolSLAM::Imu::timestamp)
        .def_readwrite("a", &PoolSLAM::Imu::a)
        .def_readwrite("w", &PoolSLAM::Imu::w)
        .def_readwrite("t", &PoolSLAM::Imu::t)
        .def("__repr__", [](const PoolSLAM::Imu& imu) {
            std::ostringstream oss;
            oss << "Imu(timestamp=" << imu.timestamp
                << ", a=[" << imu.a.x() << ", " << imu.a.y() << ", " << imu.a.z() << "]"
                << ", w=[" << imu.w.x() << ", " << imu.w.y() << ", " << imu.w.z() << "]"
                << ", t=" << imu.t << ")";
            return oss.str();
        });

    py::class_<PoolSLAM::Depth>(m, "Depth")
        .def(py::init<>())
        .def(py::init([](double timestamp, double z) {
            PoolSLAM::Depth sample{};
            sample.timestamp = timestamp;
            sample.z = z;
            return sample;
        }), py::arg("timestamp"), py::arg("z"))
        .def_readwrite("timestamp", &PoolSLAM::Depth::timestamp)
        .def_readwrite("z", &PoolSLAM::Depth::z)
        .def("__repr__", [](const PoolSLAM::Depth& d) {
            std::ostringstream oss;
            oss << "Depth(timestamp=" << d.timestamp << ", z=" << d.z << ")";
            return oss.str();
        });

    // --- State Output Struct (SystemTypes.hpp) ---
    py::class_<PoolSLAM::Pose>(m, "Pose")
        .def(py::init<>())
        .def(py::init([](double timestamp,
                         const Eigen::Vector3d& t,
                         const Eigen::Vector4d& r_wxyz,
                         const Eigen::Vector3d& v) {
            PoolSLAM::Pose pose{};
            pose.timestamp = timestamp;
            pose.t = t;
            pose.r = Eigen::Quaterniond(r_wxyz(0), r_wxyz(1), r_wxyz(2), r_wxyz(3));
            pose.v = v;
            return pose;
        }), py::arg("timestamp") = 0.0,
            py::arg("t") = Eigen::Vector3d::Zero(),
            py::arg("r") = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
            py::arg("v") = Eigen::Vector3d::Zero())
        .def_readwrite("timestamp", &PoolSLAM::Pose::timestamp)
        .def_readwrite("t", &PoolSLAM::Pose::t)
        .def_property_readonly("position", [](const PoolSLAM::Pose& p) { return p.t; })
        .def_property("r",
            [](const PoolSLAM::Pose& p) {
                return Eigen::Vector4d(p.r.w(), p.r.x(), p.r.y(), p.r.z());
            },
            [](PoolSLAM::Pose& p, const Eigen::Vector4d& q) {
                p.r = Eigen::Quaterniond(q(0), q(1), q(2), q(3));
            })
        .def_property_readonly("orientation", [](const PoolSLAM::Pose& p) {
            return Eigen::Vector4d(p.r.w(), p.r.x(), p.r.y(), p.r.z());
        })
        .def_readwrite("v", &PoolSLAM::Pose::v)
        .def_property_readonly("velocity", [](const PoolSLAM::Pose& p) { return p.v; })
        .def("__repr__", [](const PoolSLAM::Pose& p) {
            std::ostringstream oss;
            oss << "Pose(timestamp=" << p.timestamp
                << ", t=[" << p.t.x() << ", " << p.t.y() << ", " << p.t.z() << "]"
                << ", r=[w=" << p.r.w() << ", x=" << p.r.x() << ", y=" << p.r.y() << ", z=" << p.r.z() << "]"
                << ", v=[" << p.v.x() << ", " << p.v.y() << ", " << p.v.z() << "])";
            return oss.str();
        });

    // --- SystemConfig ---
    py::class_<PoolSLAM::SystemConfig>(m, "SystemConfig")
        .def(py::init<const std::string&>(), py::arg("config_file"))
        .def_readwrite("w", &PoolSLAM::SystemConfig::w)
        .def_readwrite("h", &PoolSLAM::SystemConfig::h)
        .def_readwrite("fx", &PoolSLAM::SystemConfig::fx)
        .def_readwrite("fy", &PoolSLAM::SystemConfig::fy)
        .def_readwrite("cx", &PoolSLAM::SystemConfig::cx)
        .def_readwrite("cy", &PoolSLAM::SystemConfig::cy)
        .def_readwrite("k1", &PoolSLAM::SystemConfig::k1)
        .def_readwrite("k2", &PoolSLAM::SystemConfig::k2)
        .def_readwrite("p1", &PoolSLAM::SystemConfig::p1)
        .def_readwrite("p2", &PoolSLAM::SystemConfig::p2)
        .def_readwrite("vocab_path", &PoolSLAM::SystemConfig::vocabPath)
        .def_readwrite("body_to_camera", &PoolSLAM::SystemConfig::bodyToCamera)
        .def_readwrite("body_to_depth", &PoolSLAM::SystemConfig::bodyToDepth)
        .def("__repr__", [](const PoolSLAM::SystemConfig& cfg) {
            std::ostringstream oss;
            oss << "SystemConfig(w=" << cfg.w << ", h=" << cfg.h
                << ", fx=" << cfg.fx << ", fy=" << cfg.fy
                << ", cx=" << cfg.cx << ", cy=" << cfg.cy
                << ", vocab_path='" << cfg.vocabPath << "')";
            return oss.str();
        });

    // --- System ---
    py::class_<PoolSLAM::System>(m, "System")
        .def(py::init<PoolSLAM::SystemConfig>(),
             py::arg("config"),
             py::call_guard<py::gil_scoped_release>())
        .def("start", &PoolSLAM::System::start,
             py::call_guard<py::gil_scoped_release>())
        .def("stop", &PoolSLAM::System::stop,
             py::call_guard<py::gil_scoped_release>())
        .def("push_imu", &PoolSLAM::System::pushImu,
             py::arg("imu"),
             py::call_guard<py::gil_scoped_release>())
        .def("push_depth", &PoolSLAM::System::pushDepth,
             py::arg("depth"),
             py::call_guard<py::gil_scoped_release>())
        .def("push_frame", [](PoolSLAM::System& self,
                              py::array_t<uint8_t, py::array::c_style | py::array::forcecast> frame,
                              double timestamp) {
            py::buffer_info buf = frame.request();
            const auto& cfg = self.getConfig();

            const bool valid2D = (buf.ndim == 2 &&
                                  buf.shape[0] == static_cast<py::ssize_t>(cfg.h) &&
                                  buf.shape[1] == static_cast<py::ssize_t>(cfg.w));
            const bool valid3D = (buf.ndim == 3 &&
                                  buf.shape[0] == static_cast<py::ssize_t>(cfg.h) &&
                                  buf.shape[1] == static_cast<py::ssize_t>(cfg.w) &&
                                  buf.shape[2] == 1);
            const bool valid1D = (buf.ndim == 1 &&
                                  buf.shape[0] == static_cast<py::ssize_t>(cfg.w) * static_cast<py::ssize_t>(cfg.h));

            if (!valid2D && !valid3D && !valid1D) {
                throw py::value_error(
                    "Frame must be a single-channel contiguous uint8 array of shape (" +
                    std::to_string(cfg.h) + ", " + std::to_string(cfg.w) +
                    "), (" + std::to_string(cfg.h) + ", " + std::to_string(cfg.w) +
                    ", 1), or flat size " + std::to_string(cfg.w * cfg.h));
            }

            auto* rawPixels = static_cast<uint8_t*>(buf.ptr);
            py::gil_scoped_release release;
            return self.pushFrame(rawPixels, timestamp);
        }, py::arg("frame"), py::arg("timestamp"))
        .def("get_pose", &PoolSLAM::System::getPose,
             py::call_guard<py::gil_scoped_release>())
        .def("get_config", &PoolSLAM::System::getConfig,
             py::return_value_policy::copy)
        .def("__enter__", [](PoolSLAM::System& self) -> PoolSLAM::System& {
            {
                py::gil_scoped_release release;
                self.start();
            }
            return self;
        }, py::return_value_policy::reference_internal)
        .def("__exit__", [](PoolSLAM::System& self,
                            const py::object&,
                            const py::object&,
                            const py::object&) {
            py::gil_scoped_release release;
            self.stop();
        });
}