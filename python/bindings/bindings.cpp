#include <cstdint>
#include <sstream>
#include <string>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pool_slam/System.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pool_slam, m) {
    m.doc() = "Python bindings for PoolSLAM";

    // --- Sensor Ingestion Structs (HalTypes.hpp) ---
    py::class_<PoolSLAM::Imu>(m, "Imu")
        .def(py::init<>())
        .def(py::init([](decltype(PoolSLAM::Imu::t) t,
                         const decltype(PoolSLAM::Imu::a)& a,
                         const decltype(PoolSLAM::Imu::w)& w) {
            PoolSLAM::Imu sample{};
            sample.t = t;
            sample.a = a;
            sample.w = w;
            return sample;
        }), py::arg("t"), py::arg("a"), py::arg("w"))
        .def_readwrite("t", &PoolSLAM::Imu::t)
        .def_readwrite("a", &PoolSLAM::Imu::a)
        .def_readwrite("w", &PoolSLAM::Imu::w)
        .def("__repr__", [](const PoolSLAM::Imu& imu) {
            std::ostringstream oss;
            oss << "Imu(t=" << imu.t
                << ", a=[" << imu.a.x() << ", " << imu.a.y() << ", " << imu.a.z() << "]"
                << ", w=[" << imu.w.x() << ", " << imu.w.y() << ", " << imu.w.z() << "])";
            return oss.str();
        });

    py::class_<PoolSLAM::Depth>(m, "Depth")
        .def(py::init<>())
        .def(py::init([](decltype(PoolSLAM::Depth::t) t,
                         decltype(PoolSLAM::Depth::z) z) {
            PoolSLAM::Depth sample{};
            sample.t = t;
            sample.z = z;
            return sample;
        }), py::arg("t"), py::arg("z"))
        .def_readwrite("t", &PoolSLAM::Depth::t)
        .def_readwrite("z", &PoolSLAM::Depth::z)
        .def("__repr__", [](const PoolSLAM::Depth& d) {
            std::ostringstream oss;
            oss << "Depth(t=" << d.t << ", z=" << d.z << ")";
            return oss.str();
        });

    // --- State Output Struct (SystemTypes.hpp) ---
    // Populate with the exact member fields defined in SystemTypes.hpp
    py::class_<PoolSLAM::Pose>(m, "Pose")
        .def(py::init<>());
        // .def_readonly("t", &PoolSLAM::Pose::t)
        // .def_readonly("position", &PoolSLAM::Pose::position)
        // .def_readonly("orientation", &PoolSLAM::Pose::orientation);

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
            const bool valid1D = (buf.ndim == 1 &&
                                  buf.shape[0] == static_cast<py::ssize_t>(cfg.w) * cfg.h);

            if (!valid2D && !valid1D) {
                throw py::value_error(
                    "Frame must be a contiguous uint8 array of shape (" +
                    std::to_string(cfg.h) + ", " + std::to_string(cfg.w) +
                    ") or flat size " + std::to_string(cfg.w * cfg.h));
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