from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import platform
import shutil

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + os.sys.executable]
        cuda_host_compiler = self._resolve_cuda_host_compiler()
        if cuda_host_compiler:
            cmake_args.append('-DCMAKE_CUDA_HOST_COMPILER=' + cuda_host_compiler)

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.', '--target', ext.name] + build_args, cwd=self.build_temp)

    @staticmethod
    def _resolve_cuda_host_compiler():
        # Highest priority: explicit user choice.
        explicit = os.environ.get('CUDAHOSTCXX')
        if explicit:
            return explicit

        # On Linux ARM64, nvcc from distro CUDA often fails with g++-13 system headers.
        # Prefer older host compilers when they are present.
        if platform.system() == 'Linux' and platform.machine() in ('aarch64', 'arm64'):
            for candidate in ('g++-12', 'g++-11'):
                compiler = shutil.which(candidate)
                if compiler:
                    return compiler

        return None

setup(
    name='cuTensor',
    version='0.1',
    author='Roberto Paredes',
    description='Python bindings for cuTensor C++/CUDA library',
    long_description='',
    ext_modules=[CMakeExtension('cuTensor', '.')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Operating System :: OS Independent',
    ],
)
