#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cutensor.h" 
namespace py = pybind11;

string version("0.1");

static std::vector<py::ssize_t> to_py_shape(const tshape &shape) {
    std::vector<py::ssize_t> py_shape;
    py_shape.reserve(shape.size());
    for (int dim : shape) {
        py_shape.push_back(static_cast<py::ssize_t>(dim));
    }
    return py_shape;
}


// To apply a python function to the tensor.
void cuTensor::apply(py::function func, py::args args, py::kwargs kwargs) {
        py::array_t<float> data_array(to_py_shape(shape));
        gpu_copy_from(device, size, ptr, static_cast<float *>(data_array.mutable_data()));

        // Call the provided function on the CPU data
        func(data_array, *args, **kwargs);   

        gpu_copy_to(device, size, (float *)data_array.mutable_data(), ptr);
}

 py::array_t<float>cuTensor::to_numpy() {
        py::array_t<float> data_array(to_py_shape(shape));
        gpu_copy_from(device, size, ptr, static_cast<float *>(data_array.mutable_data()));
        return data_array;
}

// create tensor from numpy array
static cuTensor* from_numpy(const py::array_t<float, py::array::c_style | py::array::forcecast>& arr, const int device=0, const string name="") {
        py::buffer_info buf = arr.request();
        std::vector<int> shape;
        for (auto dim : buf.shape) {
            shape.push_back(static_cast<int>(dim));
        }
        float *ptr = static_cast<float *>(buf.ptr);
        return new cuTensor(shape, ptr, device, name);
}

// create a tensor from a numpy array loaded from a file
static cuTensor* from_file(const string filename, const int device=0, const string name="") {
    py::array_t<float> arr = py::reinterpret_borrow<py::array_t<float>>(py::module::import("numpy").attr("load")(filename));
    return from_numpy(arr, device, name);
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
        .def("squeeze", &cuTensor::squeeze)       
        .def("unsqueeze", &cuTensor::unsqueeze)
        .def("permute", &cuTensor::permute)
        .def("apply", &cuTensor::apply)
        .def("to_numpy", &cuTensor::to_numpy)

        // static
        .def_static("mm", &cuTensor::mult2D)
        .def_static("mm_out", [](cuTensor &a, cuTensor &b, cuTensor &out) {
            cuTensor::mult2D_out(&a, &b, &out);
        })
        .def_static("from_numpy", from_numpy, py::arg("array"), py::arg("device")=0, py::arg("name")="")
        .def_static("from_file", from_file, py::arg("filename"), py::arg("device")=0, py::arg("name")="")


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
                throw py::value_error("transpose only for 2D Tensor, use permute instead");
            }
            return self.permute({1,0});
        })
        
        // Properties
        .def_property_readonly("dim", &cuTensor::getDim)
        .def_property_readonly("device", &cuTensor::getDevice)
        .def_property_readonly("shape", &cuTensor::getShape)
        .def_property_readonly("stride", &cuTensor::getStride)
        .def_property_readonly("size", &cuTensor::getSize)
        .def_property_readonly("name", &cuTensor::getName)

        // operator overloading
        .def("__str__", &cuTensor::tostr)
        .def("__len__", &cuTensor::getSize)
        .def("__invert__",[](cuTensor &self) {
            if (self.get_ndim()!=2){
                throw py::value_error("transpose only for 2D Tensor, use permute instead");
            }
            return self.permute({1,0});
        })
        .def("__add__", [](cuTensor& t1, cuTensor& t2) {
            cuTensor *t = cuTensor::sum(&t1, &t2);
            return t;
        })
        .def("__sub__", [](cuTensor& t1, cuTensor& t2) {
            cuTensor *neg = cuTensor::mult(&t2, -1);
            cuTensor *out = cuTensor::sum(&t1, neg);
            delete neg;
            return out;
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
        .def("__matmul__", [](cuTensor& t1, cuTensor& t2) {
            cuTensor *t = cuTensor::mult2D(&t1, &t2);
            return t;
        })
        .def("__truediv__", [](cuTensor& t1, float s) {
            cuTensor *t = cuTensor::mult(&t1, 1.0/s);
            return t;
        }) 
        .def("__truediv__", [](cuTensor& t1, cuTensor& t2) {
            cuTensor *aux = t2.inv();
            cuTensor *t = cuTensor::elementwise_product(&t1, aux);
            delete aux;
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
