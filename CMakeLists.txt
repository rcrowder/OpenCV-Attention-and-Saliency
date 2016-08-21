cmake_minimum_required(VERSION 2.8) 
project(SaliencyTest)

find_package(OpenCV REQUIRED)

set(NAME_SRC
    src/main.cpp
    src/BMS.cpp
)

set(NAME_HEADERS       
     inc/BMS.h
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

add_executable(SaliencyTest ${NAME_SRC} ${NAME_HEADERS})
target_link_libraries(SaliencyTest ${OpenCV_LIBS})
