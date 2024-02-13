#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cutensor.h" // Adjust this include path as necessary

namespace py = pybind11;

PYBIND11_MODULE(cuTensor, m) {  
    // Call gpu_init() directly here to ensure it runs on module import
    gpu_init();

    m.def("gpu_init", &gpu_init); // If you also want to expose it to Python

    py::class_<cuTensor>(m, "cuTensor")
        .def(py::init<>())
        .def(py::init<const std::vector<int>&, int, float*>(), py::arg("shape"), py::arg("device") = 0, py::arg("cpu_ptr") = nullptr)
        .def(py::init<const std::vector<int>&>())
        .def(py::init<const std::vector<int>&, float*>())
        .def(py::init<const std::vector<int>&, int>())
        .def("clone", &cuTensor::clone)
        .def("fill", &cuTensor::fill)
        .def("info", &cuTensor::info)
        .def("print", &cuTensor::print)
        .def("reshape", &cuTensor::reshape)
        .def_static("sum", &cuTensor::sum)
        .def_static("mult2D", &cuTensor::mult2D);
        
}
