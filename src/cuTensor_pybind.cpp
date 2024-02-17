#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cutensor.h" 
namespace py = pybind11;

string version("0.1");


// To apply a python function to the tensor
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


// create tensor from numpy array
static cuTensor* from_numpy(const py::array_t<float>& arr, const int device=0, const string name="") {
        py::buffer_info buf = arr.request();
        std::vector<int> shape;
        for (auto dim : buf.shape) {
            shape.push_back(static_cast<int>(dim));
        }
        float *ptr = static_cast<float *>(buf.ptr);
        return new cuTensor(shape, ptr, device, name);
}



PYBIND11_MODULE(cuTensor, m) {  
    // Call gpu_init() directly to ensure it runs on module import
    gpu_init(); 

    
    //m.def("gpu_init", &gpu_init); // If you also want to expose it to Python
    m.def("hw_info", &hw_info);
    m.attr("__version__") = version;
    
    py::class_<cuTensor>(m, "cuTensor")
        .def(py::init<>())
        
        // Constructor with shape, device, and name arguments
        .def(py::init<const tshape&, int, string>(),
            py::arg("shape"), py::arg("device") = 0, py::arg("name") = std::string())
                
        .def("fill", (void (cuTensor::*)()) &cuTensor::fill)
        .def("fill", (void (cuTensor::*)(float)) &cuTensor::fill, py::arg("value"))
        .def("print_array", &cuTensor::info)
        .def("print", &cuTensor::print)
        .def("reshape", &cuTensor::reshape)       
        .def("permute", &cuTensor::permute)
        .def("apply", &cuTensor::apply)

        // static
        .def_static("mm", &cuTensor::mult2D)
        .def_static("from_numpy", from_numpy, py::arg("array"), py::arg("device")=0, py::arg("name")="")
        .def_static("from_array", from_numpy, py::arg("array"), py::arg("device")=0, py::arg("name")="")

        // Lambdas
        .def("setName", [](cuTensor& self, const std::string& n) {
            self.name = n; // Assuming 'name' is a public member of cuTensor
        })
        .def("clone", [](const cuTensor &self) {
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
            return self.permute({1,0});
        })
        
        // Properties
        .def_property_readonly("dim", &cuTensor::getDim)
        .def_property_readonly("device", &cuTensor::getDevice)
        .def_property_readonly("shape", &cuTensor::getShape)
        .def_property_readonly("stride", &cuTensor::getStride)
        .def_property_readonly("size", &cuTensor::getSize)

        // operator overloading
        .def("__str__", &cuTensor::tostr)
        .def("__len__", &cuTensor::getSize)
        .def("__invert__",[](cuTensor &self) {
            if (self.get_ndim()!=2){
                printf("transpose only for 2D Tensor, use permute instead\n");
                exit(1);
            }
            return self.permute({1,0});
        })
        .def("__add__", [](cuTensor& t1, cuTensor& t2) {
            cuTensor *t = cuTensor::sum(&t1, &t2);
            return t;
        })
        .def("__sub__", [](cuTensor& t1, cuTensor& t2) {
            cuTensor *t = cuTensor::mult(&t2, -1);
            cuTensor *t3 = cuTensor::sum(&t1, cuTensor::mult(&t2, -1));
            delete t;
            return t3;
        })
        .def("__add__", [](cuTensor& t1, float s) {
            cuTensor *t = cuTensor::sumf(&t1, s);
            return t;
        })
        .def("__radd__", [](cuTensor& t1, float s) {
            cuTensor *t = cuTensor::sumf(&t1, s);
            return t;
        })
        .def("__sub__", [](cuTensor& t1, float s) {
            cuTensor *t = cuTensor::sumf(&t1, -s);
            return t;
        })
        .def("__rsub__", [](cuTensor& t1, float s) {
            cuTensor *t = cuTensor::mult(&t1, -1);
            cuTensor *t2 = cuTensor::sumf(t, s);
            delete t;
            return t2;
        })
        .def("__neg__", [](cuTensor& t1) {
            cuTensor *t = cuTensor::mult(&t1, -1);
            return t;
        })
        .def("__mul__", [](cuTensor& t1, float s) {
            cuTensor *t = cuTensor::mult(&t1, s);
            return t;
        })
        .def("__rmul__", [](cuTensor& t1, float s) {
            cuTensor *t = cuTensor::mult(&t1, s);
            return t;
        })
        .def("__mul__", [](cuTensor& t1, cuTensor& t2) {
            cuTensor *t = cuTensor::elementwise_product(&t1, &t2);
            return t;
        })
        .def("__truediv__", [](cuTensor& t1, float s) {
            cuTensor *t = cuTensor::mult(&t1, 1.0/s);
            return t;
        })
        .def("__rtruediv__", [](cuTensor& t1, float s) {
            cuTensor *t = t1.inv();
            cuTensor *t2 = cuTensor::mult(t, s);
            delete t;
            return t2;
        })
        .def("__pow__", [](cuTensor& t1, float s) {
            cuTensor *t =t1.pow(s);
            return t;
        });

}
