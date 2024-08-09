#!/bin/bash
progress() {
        #进度条程序
        local main_pid=$1
        mark_str1="Verifying"
        mark_str2="runtime"
        mark_str3="compiler"
        mark_str4="opp"
        mark_str5="toolkit"
        mark_str6="aoe"
        mark_str7="mindstudio"
        mark_str8="test-ops"
        mark_str9="pyACL"
        mark_str10="ncs"
        while [ "$(ps -p ${main_pid} | wc -l)" -ne "1" ] ; do
                mark=$(tail -n 1 install.log)
                if [[ $mark =~ $mark_str1 ]]
                then
                    mark="校验安装包，请等待"
                elif [[ $mark =~ $mark_str2 ]]
                then
                    mark="正在安装runtime组件包，请等待"
                elif [[ $mark =~ $mark_str3 ]]
                then
                    mark="正在安装compiler组件包，请等待"
                elif [[ $mark =~ $mark_str4 ]]
                then
                    mark="正在安装opp组件包，请等待"
                elif [[ $mark =~ $mark_str5 ]]
                then
                    mark="正在安装${mark_str5}组件包，请等待"
                elif [[ $mark =~ $mark_str6 ]]
                then
                    mark="正在安装${mark_str6}组件包，请等待"
                elif [[ $mark =~ $mark_str7 ]]
                then
                    mark="正在安装${mark_str7}组件包，请等待"
                elif [[ $mark =~ $mark_str8 ]]
                then
                    mark="正在安装${mark_str8}组件包，请等待"
                elif [[ $mark =~ $mark_str9 ]]
                then
                    mark="正在安装${mark_str9}组件包，请等待"
                elif [[ $mark =~ $mark_str10 ]]
                then
                    mark="正在安装${mark_str10}组件包，请等待"
                else
                    mark="正在准备安装包，请等待"
                fi
                pool=("." ".." "..." "...." "....." "......")
                num=${#pool[*]}
                roll_mark=${pool[$((RANDOM%num))]}
                echo -ne  "\033[31m ${mark} ${roll_mark}\033[0m                    \r"
                sleep 0.5
        done
}

export no_proxy=127.0.0.1,localhost,172.16.*,iam.cn-southwest-2.huaweicloud.com,pip.modelarts.private.com
export NO_PROXY=127.0.0.1,localhost,172.16.*,iam.cn-southwest-2.huaweicloud.com,pip.modelarts.private.com

if [ -e "Ascend-cann-toolkit_8.0.RC2.alpha003_linux-aarch64.run" ]; then
    echo "toolkit is exists, skipping the download step."
else
    echo "Start downloading toolkit package."
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C18SPC703/Ascend-cann-toolkit_8.0.RC2.alpha003_linux-aarch64.run -O Ascend-cann-toolkit_8.0.RC2.alpha003_linux-aarch64.run
    chmod +x Ascend-cann-toolkit_8.0.RC2.alpha003_linux-aarch64.run >/dev/null 2>&1
fi
if [ ! -d "/home/ma-user/Ascend/ascend-toolkit/8.0.RC2.alpha003" ]; then
    ./Ascend-cann-toolkit_8.0.RC2.alpha003_linux-aarch64.run --install --force --quiet >install.log 2>&1 &
    do_sth_pid=$(jobs -p | tail -1)
    progress "${do_sth_pid}" &
    wait "${do_sth_pid}"
    printf "\033[32m CANN包部署完成                         \033[0m\n"
else
    printf "\033[32m CANN包已部署                         \033[0m\n"
fi

source /home/ma-user/Ascend/ascend-toolkit/set_env.sh
echo "source /home/ma-user/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc

if [ -d "cmake-3.28.3-linux-aarch64" ]; then
    echo "The cmake folder exists, skipping the download and decompression steps."
elif [ -e "cmake-3.28.3-linux-aarch64.tar.gz" ]; then
    echo "CMake compressed file exists, start decompressing steps."
    tar xf cmake-3.28.3-linux-aarch64.tar.gz
else
    echo "need CMake compressed file exists."
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/AscendC/ResourceDependent/cmake-3.28.3-linux-aarch64.tar.gz
    tar xf cmake-3.28.3-linux-aarch64.tar.gz
fi
export PATH=/home/ma-user/work/cmake-3.28.3-linux-aarch64/bin:$PATH
echo 'export PATH=/home/ma-user/work/cmake-3.28.3-linux-aarch64/bin:$PATH' >> ~/.bashrc

