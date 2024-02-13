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
        .def(py::init<const std::vector<int>&, int, string>(),
            py::arg("shape"), py::arg("device") = 0,py::arg("name") = std::string())
        .def("fill", &cuTensor::fill)
        .def("info", &cuTensor::info)
        .def("print", &cuTensor::print)
        .def("reshape", &cuTensor::reshape)
        .def_static("sum", &cuTensor::sum)
        .def_static("mult2D", &cuTensor::mult2D)
        
        // Lambdas
        .def("setName", [](cuTensor& self, const std::string& n) {
            self.name = n; // Assuming 'name' is a public member of cuTensor
        })
        .def("clone", [](const cuTensor &self) {
            // This lambda matches the original clone behavior
            return self.clone();
        })
        .def("clone", [](cuTensor& self, const std::string& n) {
            cuTensor *t=self.clone();
            t->name = n;
            return t;
        });
}
