#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cutensor.h" // Adjust this include path as necessary

namespace py = pybind11;
// Define your function type

void cuTensor::apply(py::function func, py::args args, py::kwargs kwargs) {
        // Copy tensor data to CPU memory
        float *cpu_data = new float[size];
        //info();
        gpu_copy_from(device, size, ptr, cpu_data);

        // Convert the CPU data to a NumPy array
        py::array_t<float> data_array({size}, cpu_data);

        // Call the provided function on the CPU data
        func(data_array, *args, **kwargs);   

        gpu_copy_to(device, size, (float *)data_array.mutable_data(), ptr);

        delete[] cpu_data;

    }

PYBIND11_MODULE(cuTensor, m) {  
    // Call gpu_init() directly here to ensure it runs on module import
    gpu_init();

    //m.def("gpu_init", &gpu_init); // If you also want to expose it to Python
    m.def("hw_info", &hw_info);
    

    py::class_<cuTensor>(m, "cuTensor")
        .def(py::init<>())
        .def(py::init<const std::vector<int>&, int, string>(),
            py::arg("shape"), py::arg("device") = 0,py::arg("name") = std::string())
        .def("fill", (void (cuTensor::*)()) &cuTensor::fill)
        .def("fill", (void (cuTensor::*)(float)) &cuTensor::fill, py::arg("value"))
        .def("info", &cuTensor::info)
        .def("print", &cuTensor::print)
        .def("reshape", &cuTensor::reshape)       
        .def("permute", &cuTensor::permute)
        .def("apply", &cuTensor::apply)
        // static
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
        })
        .def("transpose",[](cuTensor &self) {
            if (self.get_ndim()!=2){
                printf("transpose only for 2D Tensor, use permute instead\n");
                exit(1);
            }
            self.permute({1,0});
        })
        
        // Properties
        .def_property_readonly("dim", &cuTensor::getDim)
        .def_property_readonly("device", &cuTensor::getDevice)
        .def_property_readonly("shape", &cuTensor::getShape)
        .def_property_readonly("stride", &cuTensor::getStride)
        .def_property_readonly("size", &cuTensor::getSize);

}
