import streamlit as st
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def main():
    st.title("Sample Size Study Calculator")
    st.write("Select the parameters for the study and the calculator will return the sample size required for the study.")

    # Add selection box for trial type
    trial_type = st.selectbox("Select trial type", ["Non-inferiority trial", "Equality", "Superiority", "Equivalence"])

    with st.expander("Get treatment failure rate from Tai, 2022 study"):
        img = Image.open(os.path.join(os.getcwd(), 'ofac363f1.jpg'))
        img = np.array(img)

        col1_ex, col2_ex = st.columns(2)
        year = col1_ex.number_input("Study timepoint (years)", value = 1., step = 0.5, format = "%.1f")


        pixel_horz = pixel_from_year(img, year)
        pixel_vert = get_height_location(img, pixel_horz)

        incidence_plot = col2_ex.markdown(f'''Incidence of treatment failure  
                                                **{get_incidence(pixel_vert):.2f}**''')

        zero_coords = [69, 434]

        study_img, ax = plt.subplots()
        ax.imshow(img)
        ax.plot(pixel_horz, pixel_vert, 'rx')
        ax.hlines(pixel_vert, zero_coords[0], pixel_horz, colors='r', alpha = 0.5)
        ax.vlines(pixel_horz, zero_coords[1], pixel_vert, colors='r', alpha = 0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        # display figure in streamlit
        st.pyplot(study_img)
        
    
    # Add two columns for alpha and power inputs
    st.subheader("Statistical Parameters")
    col1, col2 = st.columns(2)

    # Add input box for confidence level (alpha) in the first column
    alpha = col1.number_input("Enter confidence level (alpha)", value=0.05, step=0.001, format="%.3f")
    col1.markdown(f'''Confidence level (alpha): **{alpha:.3f}** ''') # Display alpha with 3 decimal places

    # Add input box for power in the second column
    power = col2.number_input("Enter power (%)", value=80, step=1, format="%i")
    col2.markdown(f'''Power: **{power} %** ''') # Display power with 3 decimal places

    st.subheader("Study Parameters")
    # Add two columns for reference and experimental treatment success rates
    col1, col2 = st.columns(2)
    reference_success = col1.number_input("Enter reference treatment success rate (%)", value=50., step=1., format="%.2f")
    col1.markdown(f'''Reference treatment success rate: **{reference_success:.2f} %** ''')
    experimental_success = col2.number_input("Enter experimental treatment success rate (%)", value=50., step=1., format="%.2f")
    col2.markdown(f'''Experimental treatment success rate: **{experimental_success:.2f} %** ''')

    test_margin = col1.number_input("Enter absolute non-inferiority limit", value=10., step=1., format="%.2f")
    col1.markdown(f'''Test margin: **{test_margin:.2f} %** ''')

    st.header("Sample Size Results")
    if trial_type == "Non-inferiority trial":
        sample_size = get_sample_size_non_inf(reference_success/100, experimental_success/100, alpha, power, test_margin/100)
    elif trial_type == "Equality":
        sample_size = get_sample_size_equality(reference_success/100, experimental_success/100, alpha, power)
    elif trial_type == "Superiority":
        sample_size = get_sample_size_sup(reference_success/100, experimental_success/100, alpha, power, test_margin/100)
    elif trial_type == "Equivalence":
        sample_size = get_sample_size_equivalence(reference_success/100, experimental_success/100, alpha, power, test_margin/100)

    st.markdown(f" ### Sample size per trial arm: **{np.ceil(sample_size)}**")
    st.subheader(f"Sample size total for both trial arms: **{np.ceil(sample_size) * 2}**")

    st.markdown(f'''We assume that the **{year}** year treatment success rates will be **{reference_success} %** for the control group and **{experimental_success} %** for the intervention group. We have set a one-sided significance level of **{alpha}** to test our null hypothesis and aim for a power of **{power} %** to detect that the treatment is truly not inferior to the control, based on a **{test_margin} %** margin. Given these parameters, we estimate that a total sample size of **{round(sample_size)}** patients will be required per arm to adequately power this study''')


def pixel_from_year(img:np.array, year:float) -> int:
    # get location of black pixels along a width of an image
    black_pixels = np.where(img[450, :, 0] <= 30)[0]
    year_pixels = round(np.mean(black_pixels[1:] - black_pixels[:-1]))
    # print(f'year_pixel_distance: {year_pixels}')

    return int(year_pixels * year + black_pixels[0])

def get_height_location(img:np.array, horz_pixel_location: int) -> int:
    color_pixels = np.where((img[:453, horz_pixel_location, 0] >= 130) & (img[:453, horz_pixel_location, 0] < 250) &
                            (img[:453, horz_pixel_location, 1] < 180) & (img[:453, horz_pixel_location, 2] < 180) &
                            (img[:453, horz_pixel_location, 0] != img[:453, horz_pixel_location, 1]))[0]

    return round(np.mean(color_pixels))
    

def get_incidence(height_pixels:int) -> float:
    inc_values = {0:434, 10:349, 20:263, 30:178, 40:92}
    cal_pixels = [*inc_values.values()][::-1]
    cal_values = [*inc_values.keys()][::-1]
    # print(height_pixels)

    # get the incidence value from the image
    
    return np.interp(height_pixels, cal_pixels, cal_values)

def get_sample_size_non_inf(p_reference, p_experimental, alpha, power, bound):
    '''
    Calculate the sample size for a given power, alpha, and bound
    '''
    ratio_power = power / 100
    beta = 1 - ratio_power
    zb = stats.norm.ppf(beta)
    za = stats.norm.ppf(1 - (alpha))
    # if (p_reference - p_experimental - bound) == 0:
    #     return np.inf
    sample_size = ((za - zb)**2 * (p_reference * (1 - p_reference) + p_experimental * (1 - p_experimental))) / (p_reference - p_experimental - bound)**2

    return sample_size

def get_sample_size_equality(p_reference, p_experimental, alpha, power):
    '''
    Calculate the sample size for a given power, alpha, and bound
    '''
    ratio_power = power / 100
    beta = 1 - ratio_power
    zb = stats.norm.ppf(beta)
    za = stats.norm.ppf(1 - (alpha / 2))
    # if (p_reference - p_experimental) == 0:
    #     return np.inf
    sample_size = ((za - zb)**2 * (p_reference * (1 - p_reference) + p_experimental * (1 - p_experimental))) / (p_reference - p_experimental)**2

    return sample_size

def get_sample_size_equivalence(p_reference, p_experimental, alpha, power, bound):
    '''
    Calculate the sample size for a given power, alpha, and bound
    '''
    ratio_power = power / 100
    beta = 1 - ratio_power
    zb = stats.norm.ppf(beta/2)
    za = stats.norm.ppf(1 - (alpha))
    # if (p_reference - p_experimental - bound) == 0:
    #     return np.inf
    sample_size = ((za - zb)**2 * (p_reference * (1 - p_reference) + p_experimental * (1 - p_experimental))) / (bound)**2

    return sample_size

def get_sample_size_sup(p_reference, p_experimental, alpha, power, bound):
    '''
    Calculate the sample size for a given power, alpha, and bound
    '''
    ratio_power = power / 100
    beta = 1 - ratio_power
    zb = stats.norm.ppf(beta)
    za = stats.norm.ppf(1 - (alpha / 2))
    # if (p_reference - p_experimental - bound) == 0:
    #     return np.inf
    sample_size = ((za - zb)**2 * (p_reference * (1 - p_reference) + p_experimental * (1 - p_experimental))) / (p_reference - p_experimental)**2

    return sample_size
    
if __name__ == "__main__":
    main()
