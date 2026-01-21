import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False # Fix minus sign display issue