## Building with Conan
```shell
pip install conan
mkdir build
conan install conanfile.txt --build=missing
cd build
cmake -G"Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=./Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release ..

```