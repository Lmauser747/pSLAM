set -e
BUILD_TYPE=${1:-Release}
mkdir -p build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
ninja