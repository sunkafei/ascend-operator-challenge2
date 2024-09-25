#!/bin/bash

# 先安装
../../build_out/custom_opp_euleros_aarch64.run

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=1

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd $CURRENT_DIR

# 导出环境变量
SHORT=v:,
LONG=dtype:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while :
do
    case "$1" in
        # float16, float, int32
        (-v | --dtype)
            DTYPE="$2"
            shift 2;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

if [ ! $ASCEND_HOME_DIR ]; then
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        export ASCEND_HOME_DIR=$HOME/Ascend/ascend-toolkit/latest
    else
        export ASCEND_HOME_DIR=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
source $ASCEND_HOME_DIR/bin/setenv.bash

export DDK_PATH=$ASCEND_HOME_DIR
arch=$(uname -m)
export NPU_HOST_LIB=$ASCEND_HOME_DIR/${arch}-linux/lib64

function main {
    # 1. 清除算子输出和日志文件
    
    rm -rf ./input/*
    rm -rf ./output/*
    # rm ./input/*.bin
    # rm -rf ./output/output*.bin > /dev/null

    # 2. 生成或复用输入数据和真值数据 
    if [ -d "./input" ]; then
        if [ "$(ls -A "./input")" ]; then
        echo "已存在测试数据"
        else
            echo "生成测试数据"
            cd $CURRENT_DIR
            python3 scripts/gen_data.py
        fi
    else
        echo "生成测试数据"
        cd $CURRENT_DIR
        python3 scripts/gen_data.py
    fi

    if [ $? -ne 0 ]; then
        echo "ERROR: generate input data failed!"
        return 1
    fi
    echo "INFO: generate input data success!"

    # 3. 编译或复用acl可执行文件
    #if [ -e "./output/execute_op" ]; then
    #    echo "可执行存在"
    #else
        echo "可执行不存在"
        cd $CURRENT_DIR; rm -rf build; mkdir -p build; cd build
        cmake ../src
        if [ $? -ne 0 ]; then
            echo "ERROR: cmake failed!"
            return 1
        fi
        echo "INFO: cmake success!"
        make
        if [ $? -ne 0 ]; then
            echo "ERROR: make failed!"
            return 1
        fi
        echo "INFO: make success!"
    #fi

    # 4. 运行可执行文件
    cd $CURRENT_DIR/output
    echo "INFO: execute op!"
    timeout 30 msprof --application="execute_op" --output=./

    if [ $? -ne 0 ]; then
        echo "ERROR: acl executable run failed! please check your project!"
        return 1
    fi
    echo "INFO: acl executable run success!"

    time_ust=$(awk -F, '{print $(NF-37)}' $(find ./ -name op_summary*.csv) | tail -n 1)
    time_base=167
    time_ust=$(printf "%.0f" $time_ust)   
    echo $time_ust

    # 5. 比较真值文件
    cd $CURRENT_DIR
    ret=`python3 scripts/verify_result.py output/output.bin output/golden.bin`
    echo $ret
    if [ "x$ret" == "xtest pass" ]; then
        echo ""
        echo "#####################################"
        echo "INFO: you have passed the Precision!"
        echo "#####################################"
        echo ""
    fi
}

main
