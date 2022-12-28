
def main():
    # read data from result_yuv_1223.txt
    result = read_result_from_text('result_yuv_1223.txt')
    #print(*result['given_checkpoint'], sep="\n")

    given_checkpoint = result['given_checkpoint']
    default_checkpoint = result['default_checkpoint']

    psnr_sum = 0
    ssim_sum = 0
    for data_name in given_checkpoint.keys():
        psnr = given_checkpoint[data_name][0]
        ssim = given_checkpoint[data_name][1]
        data_num = given_checkpoint[data_name][2]

        psnr_sum += psnr * data_num
        ssim_sum += ssim * data_num

    psnr_avg = psnr_sum / sum([given_checkpoint[data_name][2] for data_name in given_checkpoint.keys()])
    ssim_avg = ssim_sum / sum([given_checkpoint[data_name][2] for data_name in given_checkpoint.keys()])

    print("given psnr_avg: ", psnr_avg, " ssim_avg: ", ssim_avg)
    print("===================================================================================================")

    psnr_sum = 0
    ssim_sum = 0
    for data_name in default_checkpoint.keys():
        psnr = default_checkpoint[data_name][0]
        ssim = default_checkpoint[data_name][1]
        data_num = default_checkpoint[data_name][2]

        psnr_sum += psnr * data_num
        ssim_sum += ssim * data_num

    psnr_avg = psnr_sum / sum([default_checkpoint[data_name][2] for data_name in default_checkpoint.keys()])
    ssim_avg = ssim_sum / sum([default_checkpoint[data_name][2] for data_name in default_checkpoint.keys()])
    print("default psnr_avg: ", psnr_avg, " ssim_avg: ", ssim_avg)

def read_result_from_text(path):
    """
    path: path to store the result
    """
    result = {}
    result["given_checkpoint"] = {}
    result["default_checkpoint"] = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "data_name" not in line:
                continue
            line_str = line 
            line = line.split(",")
            data_name = line[0].split(":")[1].strip()
            psnr = float(line[1].split(":")[1].strip())
            ssim = float(line[2].split(":")[1].strip())
            data_num = int(line[3].split()[1].strip())
            
            if "[given]" in line_str:
                checkpoint = "given_checkpoint"
            elif "[default]" in line_str:
                checkpoint = "default_checkpoint"
            else:
                print("data_name is not correct")
                raise ValueError("data_name is not correct")
            
            result[checkpoint].update({
                data_name: [psnr, ssim, data_num]
                })
            
    

    return result


def calculate(data_num, psnr_list, ssime_list):
    #  check if data_num is same
    if len(data_num) != len(psnr_list) or len(data_num) != len(ssime_list):
        print("data_num, psnr_list, ssime_list are not same length")
        return
    
    # calculate average of psnr and ssim
    psnr_avg = 0
    ssim_avg = 0
    for i in range(len(data_num)):
        psnr_avg += psnr_list[i] * data_num[i]
        ssim_avg += ssime_list[i] * data_num[i]
    psnr_avg /= sum(data_num)
    ssim_avg /= sum(data_num)

    print("psnr_avg: ", psnr_avg, " ssim_avg: ", ssim_avg)

if __name__ == "__main__":
    main()