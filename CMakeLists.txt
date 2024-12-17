cmake_minimum_required(VERSION 3.10)

# Set CUDA architectures policy
cmake_policy(SET CMP0104 NEW)

project(cumnist LANGUAGES C CUDA)

# Enable CUDA
enable_language(CUDA)

# Create executable from your source files
add_executable(cumnist 
    mnist.cu
)
# 86 = RTX 3000 series (Ampere)
set_property(TARGET cumnist PROPERTY CUDA_ARCHITECTURES 86)
