from PIL import Image as image
import util as ut

# ---!   open and create a grey scale img    !--- #
img_grey = (image.open("greyScaleSahar.jpg")).convert("L")
width, height = img_grey.size
pixels = img_grey.load()

# ---!   loop partition levels   !--- #
for d in range(1,9):
    # ---!   calc the number of partitions and ready the data-bases   !--- #
    N = pow(2, d)
    histogram = [[[0 for _ in range(256)] for _ in range(N)] for _ in range(N)]
    pdf, medians, averages = ([[[] for _ in range(N)] for _ in range(N)], [[0 for _ in range(N)] for _ in range(N)],
                              [[0 for _ in range(N)] for _ in range(N)]) #  median and averages are actually J sombrero and J hat in the assignment

    reg_width, reg_height = width // N, height // N
    width_partition, height_partition = [0], [0]
    # ---!   loop the image by areas and calculate needed info   !--- #
    for i in range(1,N+1):
        # ---!   calculate the partition on the x-axis   !--- #
        width_partition.append(i * reg_width)
        for j in range(1, N + 1):
            # ---!   calculate the partition on the y-axis   !--- #
            height_partition.append(j * reg_height)
            for x in range(width_partition[i - 1], width_partition[i]):
                for y in range(height_partition[j - 1], height_partition[j]):
                    # ---!   calculate the specific area histogram   !--- #
                    histogram[i-1][j-1][pixels[x,y]] += 1
            # ---!   calc pdf and median from histogram    !--- #
            pdf[i-1][j-1] = [rep / (reg_width * reg_height) for rep in histogram[i-1][j-1]]
            medians[i-1][j-1] = ut.find_median(histogram[i-1][j-1])
            # ---!   calc avrage from pdf    !--- #
            averages[i-1][j-1] = ut.find_averages(pdf[i-1][j-1])

    # ---!   ready the output images    !--- #
    median_output_image = image.new("L", (width, height))
    out_median_img_pixels = median_output_image.load()
    average_output_image = image.new("L", (width, height))
    out_average_img_pixels = average_output_image.load()

    # ---!   loop the image and reconstruct the images using the J's  !--- #
    mse, mad = 0, 0
    for i in range(1,N+1):
        for x in range(width_partition[i - 1], width_partition[i]):
            for j in range(1, N + 1):
                for y in range(height_partition[j - 1], height_partition[j]):
                    out_median_img_pixels[x,y] = medians[i-1][j-1]
                    out_average_img_pixels[x, y] = averages[i-1][j-1]
                    # ---!   calc the relevant errors    !--- #
                    mse += pow(pixels[x,y]-out_average_img_pixels[x,y],2)
                    mad += abs(pixels[x,y]-out_median_img_pixels[x,y])

    # ---!   save images    !--- #
    median_output_image.save("Sampling_Median/output_part2_median" + str(d) + ".png")
    average_output_image.save("Sampling_Average/output_part2_average" + str(d) + ".png")

    # ---!   display final errors    !--- #
    mse = mse/ (width*height)
    mad = mad/ (width*height)
    print("On D = " + str(d) + ": MSE is equals " + str(mse) + ", MAD equals: " + str(mad))


