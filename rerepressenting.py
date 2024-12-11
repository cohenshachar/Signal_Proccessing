from PIL import Image as image
import numpy as np
import util as ut


# ---!   open and create a grey scale img    !--- #
img_grey = (image.open("greyScaleImage512by512_op4.gif")).convert("L")
# ---!   create histogram of grey scale values from img    !--- #
width, height = img_grey.size
pixels = img_grey.load()
histogram = [0] * 256
for x in range(width):
    for y in range(height):
        histogram[pixels[x, y]] += 1

# ---!   calc pdf from histogram    !--- #
pdf = [rep / (width*height) for rep in histogram]

# ---!   calc phi's   !---#
histo_arr = np.array(histogram)
phi_h = np.max(np.nonzero(histo_arr))+1
phi_l = np.min(np.nonzero(histo_arr))
phi_delta = phi_h - phi_l
# print(phi_h, phi_l)

# print(pdf_mse)

# ---!   create decision and representation levels for 1,...,8 bits per pixel   !---#
max_lloyd_mse = []
uniform_mse = []
bits =  [b for b in range(1, 9)]
for b in bits:
    k = pow(2, b)
    # ---!   interval size, r's, d's calculated for uniform distribution   !---#
    interval_size = phi_delta / k
    u_d_levels,ml_d_levels,u_r_levels,ml_r_levels =[0] ,[0], [], []
    for i in range(1,k+1):
        u_r_levels.append(phi_l + ((i - 0.5) * interval_size))
        u_d_levels.append(phi_l + (i * interval_size))
        ml_r_levels.append(phi_l + ((i - 0.5) * interval_size))
        ml_d_levels.append(phi_l + (i * interval_size))
    ut.plot(u_d_levels,u_r_levels,"Quantizing_Uniform/uniform"+str(b))
    # ---!   setting output image   !---#
    output_image = image.new("L", (width, height))
    out_img_pixels = output_image.load()

    # ---!   uniform distribution output ahead   !---#
    uniform_mse.append(0)
    for x in range(width):
        for y in range(height):
            # ---!  the formula does not work for: x=phi_h as:
            # ---!  floor(phi_h-phi_l / interval) = (phi_h-phi_l / phi_h-phi_l/k) = floor(k) = k  !---#
            d_level = ((pixels[x, y]-phi_l)//interval_size)
            if d_level == k:
                d_level -= 1    # ---! handle x=phi_h as it should be in between d_k-1 and d_k !--- #
            out_img_pixels[x, y] = int(np.round(phi_l + (d_level + 0.5) * interval_size))  # ---! assign representation !--- #
            uniform_mse[b-1] += pow((pixels[x, y] - out_img_pixels[x, y]), 2)
    uniform_mse[b-1] = uniform_mse[b-1] / (width * height)
    output_image.save("Quantizing_Uniform/output" + str(b) + ".png")
    # ---!   uniform distribution output finished   !---#

    # ---!   max-lloyd algorithm output ahead   !---#
    ml_r_levels = ut.lloyd_max(pdf, ml_d_levels, 0.01)
    ut.plot(ml_d_levels,ml_r_levels,"Quantizing_MaxLloyd/max_lloyd"+str(b))
    ut.cmp_plot(ml_d_levels,ml_r_levels,u_d_levels,u_r_levels,"Quantizing_MaxLloyd/max_lloyd vs uniform"+str(b))

    max_lloyd_mse.append(0)
    for x in range(width):
        for y in range(height):
            for i in range(1, k+1): # ---! no formula! check were pixel falls in the decision levels !--- #
                if pixels[x, y] <= ml_d_levels[i]:
                    out_img_pixels[x, y] = int(ml_r_levels[i - 1]) # ---! assign representation !--- #
                    break
            max_lloyd_mse[b-1] += pow((pixels[x, y] - out_img_pixels[x, y]), 2)
    # ---!   max-lloyd algorithm output finished   !---#

    # ---!   save images    !--- #
    output_image.save("Quantizing_MaxLloyd/output_refined" + str(b) + ".png")

    # ---!   display final errors    !--- #
    max_lloyd_mse[b-1] = max_lloyd_mse[b-1] / (width * height)
    print("Uniform MSE on b = " + str(b) + " is: " + str(uniform_mse[b-1]) + " vs Max Lloyd MSE: "+str(max_lloyd_mse[b-1]))
bits.append(9)
ut.plot(bits,max_lloyd_mse,"Quantizing_MaxLloyd/max_lloyd_mse")
ut.plot(bits,uniform_mse,"Quantizing_Uniform/Uniform_mse")
ut.cmp_plot(bits, max_lloyd_mse, bits, uniform_mse, "Quantizing_MaxLloyd/max_lloyd vs uniform mse")

