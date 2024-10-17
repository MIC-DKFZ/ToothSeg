from pathlib import Path

import numpy as np



def age_sex(ages, sexes):
    print('Males', (sexes == 'M').sum())
    print('Females', (sexes == 'F').sum() + (sexes == 'W').sum())

    ages = np.array([float(n) if n != 'None' else np.nan for n in ages])

    male_ages = ages[sexes == 'M']
    female_ages = ages[(sexes == 'F') | (sexes == 'W')]



    male_ages = male_ages[(male_ages > 0) & (male_ages < 100)]
    female_ages = female_ages[(female_ages > 0) & (female_ages < 100)]
    ages = ages[(ages > 0) & (ages < 100)]

    print(
        'Male age',
        np.nanmedian(male_ages),
        np.nanquantile(male_ages, 0.75) - np.nanquantile(male_ages, 0.25),
        min(male_ages), max(male_ages),
    )
    print(
        'Female age',
        np.nanmedian(female_ages),
        np.nanquantile(female_ages, 0.75) - np.nanquantile(female_ages, 0.25),
        min(female_ages), max(female_ages),
    )
    print(
        'age',
        np.nanmedian(ages),
        np.nanquantile(ages, 0.75) - np.nanquantile(ages, 0.25),
        min(ages), max(ages),
    )


def runs_print(runs):



    thicks = np.array([float(r) if r.strip() != 'None' else np.nan for r in runs[:, 0]])
    spacings = np.array([float(r) if r.strip() != 'None' else np.nan for r in runs[:, 1]])
    spacings[np.isnan(spacings)] = thicks[np.isnan(spacings)]
    kvps = np.array([float(r) if r.strip() != 'None' else np.nan for r in runs[:, 2]])
    mas = np.array([float(r) if r.strip() != 'None' else np.nan for r in runs[:, 3]])

    print(((spacings >= 0.12) & (spacings < 0.16)).sum())
    print(((spacings >= 0.16) & (spacings < 0.2)).sum())
    print(((spacings >= 0.2) & (spacings < 0.25)).sum())
    print(((spacings >= 0.25) & (spacings < 0.3)).sum())   
    print(((spacings >= 0.3) & (spacings < 0.4)).sum())
    print(((spacings >= 0.4)).sum())



    print(((kvps >= 70) & (kvps < 80)).sum())
    print(((kvps >= 80) & (kvps < 90)).sum())
    print(((kvps >= 90) & (kvps < 100)).sum())
    print(((kvps >= 100) & (kvps < 110)).sum())
    print(((kvps >= 110) & (kvps < 120)).sum())
    print(((kvps >= 120)).sum())
    print('ll')



    mas = mas[(mas > 0) & (mas <= 16)]
    print(((mas >= 1) & (mas < 3)).sum())
    print(((mas >= 3) & (mas < 5)).sum())
    print(((mas >= 5) & (mas < 7)).sum())
    print(((mas >= 7) & (mas < 9)).sum())
    print(((mas >= 9) & (mas < 11)).sum())
    print(((mas >= 11)).sum())





if __name__ == '__main__':
    root = Path('/home/mkaailab/Documents/CBCT/baselines/')

    ages = np.loadtxt(root / 'ages.txt', dtype=str)
    sexes = np.loadtxt(root / 'sexes.txt', dtype=str)
    age_sex(ages, sexes)


    runs = np.loadtxt(root / 'res_kvp_ma.txt', dtype=str, delimiter=';')
    runs_print(runs)


    fovs = np.loadtxt(root / 'fovs.txt', dtype=str, delimiter=';')
    heights = np.array([float(r) if r.strip() != 'None' else np.nan for r in fovs[:, 0]])
    diameters = np.array([float(r) if r.strip() != 'None' else np.nan for r in fovs[:, 1]])

    volumes = ((np.pi / 4 * diameters**2) * heights) / 1000

    print(((volumes >= 240) & (volumes < 400)).sum())
    print(((volumes >= 400) & (volumes < 500)).sum())
    print(((volumes >= 500) & (volumes < 1000)).sum())
    print(((volumes >= 1000) & (volumes < 1500)).sum())
    print(((volumes >= 1500) & (volumes < 2500)).sum())
    print(((volumes >= 2500) & (volumes < 10000)).sum())


    k = 3
