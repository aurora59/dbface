cpp_srcs := $(shell find src -name "*.cpp")
cpp_objs := $(patsubst %.cpp,%.o,$(cpp_srcs))
cpp_objs := $(subst src/,objs/,$(cpp_objs))

# 由于cpp可能与cu同名，但是不同文件，因此对于cuda的程序把cu改成cuo
cu_srcs := $(shell find src -name "*.cu")
cu_objs := $(patsubst %.cu,%.cuo,$(cu_srcs))
cu_objs := $(subst src/,objs/,$(cu_objs))
# cu_objs := $(subst src/,workspace/,$(cu_objs))

# 定义名称参数
workspace := workspace
# cubinary := cupro
binary := pro
sbinary := db_infer.so

# 这里定义头文件和连接目标，没有加-I, -L, -l，后面用foreach一次性增加
include_paths := /usr/local/cuda-11.2/include \
				 /data-nbd/lxh/lean/cuda11-cudnn8.0.4/include \
				 /data-nbd/lxh/lean/opencv-4.2.0/include/opencv4 \
				 /home/gaocy/anaconda3/envs/py38/include/python3.8/ \
				 /data-nbd/lxh/lean/TensorRT-8.0.1.6-cuda11/include/

library_paths := /usr/local/cuda-11.2/lib64 \
				 /data-nbd/lxh/lean/cuda11-cudnn8.0.4/lib64 \
				 /data-nbd/lxh/lean/opencv-4.2.0/lib \
				 /home/gaocy/anaconda3/envs/py38/lib \
				 /data-nbd/lxh/lean/TensorRT-8.0.1.6-cuda11/lib
link_librarys := cudart cudnn opencv_core opencv_imgcodecs opencv_imgproc gomp \
				 pthread cublas python3.8 cudadevrt nvinfer nvcaffe_parser nvinfer_plugin nvonnxparser

# 定义编译选项
cpp_compile_flags := -m64 -fPIC -g -O0 -std=c++17 -w -DHAS_INT8_SUPPORT=5
cu_compile_flags  := -m64 -Xcompiler -fPIC -g -O0 -std=c++11 -w -gencode=arch=compute_75,code=sm_75

# -w屏蔽警告
# makefile里有两种类型：1.字符串 2.字符串数组（空格隔开就是数组）
# 对头文件，库文件，目标，统一增加 -I, -L, -l
# foreach var.list,cmd
#	var = item
#	list = link_librarys
#	cmd = -Wl,-rpath=$(item)
#
# output = []
# for item in link_librarys:
#	output.append(f"-Wl,-rpath={item}")
# rpath = output

# -L 指定链接目标时查找的目录
# -l 指定链接时查找的目标名称，符合libname.so, -lname规则
# -I 指定编译时头文件查找目录
rpath         := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

# 合并选项
cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        := $(rpath) $(library_paths) $(link_librarys)
# culink_flags        := $(library_paths) $(link_librarys)


# 定义cpp的编译方式
# $@ 生成项
# $< 依赖项第一个
# $^ 依赖项所有
objs/%.o : src/%.cpp
	@mkdir -p $(dir $@)
	@echo Compile $<
	@g++ -c $< -o $@ $(cpp_compile_flags)

# 定义cuda文件的编译方式
objs/%.cuo : src/%.cu
	@mkdir -p $(dir $@)
	@echo Compile $<
	@nvcc -c $< -o $@ $(cu_compile_flags)

# 定义workspace/pro文件的编译
$(workspace)/$(binary) : $(cpp_objs) $(cu_objs)
	@mkdir -p $(dir $@)
	@echo Link $@
	@g++ $^ -o $@ $(link_flags)



# 定义workspace/pro文件的编译
$(workspace)/$(sbinary) : $(cpp_objs) $(cu_objs)
	@mkdir -p $(dir $@)
	@echo Link $@
	@g++ -shared $^ -o $@ $(link_flags)

# 定义pro快捷编译指令，这里只发生编译，不执行
pro : $(workspace)/$(binary)
# @strip $< 
#给文件加密
share  : $(workspace)/$(sbinary)

# 定义编译并执行指令，并且执行目录切换到workspace下
run : pro
	@cd $(workspace) && ./$(binary)

debug :
	@echo $(cpp_objs)
	@echo $(cu_objs)

clean :
	@rm -rf objs $(workspace)/$(binary)

.PHONY : clean debug run pro