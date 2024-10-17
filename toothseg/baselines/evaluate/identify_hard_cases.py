from collections import defaultdict
import json
import multiprocessing as mp
from pathlib import Path
from typing import Tuple

import nibabel
import numpy as np
from tqdm import tqdm


def has_large_fov(scan_nii: nibabel.Nifti1Image):
    spacing = np.array(scan_nii.header.get_zooms())
    shape = np.array(scan_nii.header.get_data_shape())
    large_fov = np.all((spacing * shape) > 159).item()

    return large_fov


def has_third_molar(labels: np.ndarray):
    labels = np.clip(labels.astype(int) - 4, 0, 32)
    m3_mask = (labels[labels > 0] / 8) == (labels[labels > 0] // 8)
    third_molar = np.any(m3_mask).item()

    return third_molar


def has_tooth_gap(labels: np.ndarray):
    labels = np.clip(labels.astype(int) - 4, 0, 32)
    unique_labels = np.unique(labels)[1:]

    missing_teeth = False
    if unique_labels.shape[0] == 0:
        return missing_teeth

    for i in range(0, 32, 16):
        quadrant1 = ((i + 1) <= unique_labels) & (unique_labels <= (i + 8))
        quadrant2 = ((i + 9) <= unique_labels) & (unique_labels <= (i + 16))

        labels1 = i + 8 - unique_labels[quadrant1][::-1]
        labels2 = (unique_labels[quadrant2] - 1) % 16
        labels12 = np.concatenate((labels1, labels2))
        if labels12.shape[0] <= 1:
            continue
        diffs = labels12[1:] - labels12[:-1]
        missing_teeth = missing_teeth or diffs.max().item() > 1

    return missing_teeth


def has_metal_artifact(
    scan_nii: nibabel.Nifti1Image,
    scan: np.ndarray,
    labels: np.ndarray,
):
    metal_mask = scan >= 4095
    background_mask = labels == 0

    spurious_metal_voxels = (metal_mask & background_mask).sum()
    spurious_metal_volume = spurious_metal_voxels * np.prod(spacing)

    metal_artifact = (spurious_metal_volume >= 100).item()  # mm3

    return metal_artifact


def is_edentulous(
    labels_nii: nibabel.Nifti1Image,
    labels: np.ndarray,
):
    spacing = np.array(labels_nii.header.get_zooms())

    tooth_mask = labels > 4
    tooth_volume = tooth_mask.sum() * np.prod(spacing)

    return tooth_volume < 500


def process_case(files: Tuple[Path, Path]):
    _, labels_file = files
    
    # if not labels_file.name.startswith('thomasa-'):
    #     return labels_file, {}

    labels_nii = nibabel.load(labels_file)
    labels = np.asarray(labels_nii.dataobj)
    unique_labels = np.unique(labels)

    edentulous = is_edentulous(labels_nii, labels)
    out = {
        'large_fov': not edentulous and has_large_fov(labels_nii),
        'third_molar': not edentulous and has_third_molar(labels),
        'misaligned': not edentulous and any(labels_file.stem.startswith(stem) for stem in misaligned_stems),
        'metal_artifact': not edentulous and any(labels_file.stem.startswith(stem) for stem in metal_artifact_stems),
        'restorations': not edentulous and np.any((unique_labels == 3) | (unique_labels == 4)).item(),
    }
    
    out['normal'] = (
        not out['large_fov']
        and not out['third_molar']
        and not out['misaligned']
        and not out['metal_artifact']
        and not edentulous
        and not out['restorations']
    )

    return labels_file, out

if __name__ == '__main__':
    root = Path('/mnt/diag/CBCT/tooth_segmentation/data/')

    scan_files = sorted(root.glob('Dataset164_Filtered_Classes/imagesTs/*'))
    labels_files = sorted(root.glob('Dataset164_All_Classes/labelsTs/*'))

    misaligned_stems = [
        'amalea-theoretical',
        'berta-civilian',
        'berthe-useless',
        'britt-breakable',
        'camellia-tender',
        'correna-outer',
        'devon-scientific',
        'dory-violent',
        'dotty-prior',
        'esta-entire',
        'eudora-theoretical',
        'fawne-uncomfortable',
        'ivy-blonde',
        'jamie-unfair',
        'jaquelin-main',
        'joey-breakable',
        'karylin-poised',
        'kellia-obedient',
        'kerrin-wicked',
        'laryssa-adorable',
        'leola-horizontal',
        'libbey-clinical',
        'lonni-sure',
        'lotty-widespread',
        'madelena-strange',
        'marita-conventional',
        'melessa-determined',
        'melisande-good',
        'mella-unconscious',
        'mikaela-unemployed',
        'myrna-brainy',
        'myrta-scattered',
        'perl-constitutional',
        'roanna-educational',
        'rosalyn-appalling',
        'scarlet-misleading',
        'shani-glamorous',
        'shayne-injured',
        'terrye-anxious',
        'tessa-specified',
        'violante-double',
    ]
    metal_artifact_stems = [
        'abigale-educational-flyingfish',
        'alena-competent-dinosaur',
        'amalea-close-monkey',
        'anstice-alright-dinosaur',
        'april-afraid-whippet',
        'ardelia-armed-possum',
        'auguste-external-roadrunner',
        'aurlie-noble-lynx',
        'bambi-compulsory-marlin',
        'beckie-daily-swan',
        'berta-civilian-starfish',
        'betty-national-stingray',
        'blancha-domestic-hedgehog',
        'blisse-gradual-pony',
        'bobbi-reduced-goose',
        'breena-concerned-tiglon',
        'brenda-limited-whitefish',
        'calida-double-lamprey',
        'camellia-tender-moth',
        'camila-blind-pelican',
        'carlina-acceptable-whippet',
        'carlota-gross-grasshopper',
        'casie-ashamed-worm',
        'celle-supreme-antelope',
        'chandal-sore-takin',
        'charlean-psychological-marsupial',
        'christine-tremendous-deer',
        'connie-husky-fish',
        'cynthia-alone-giraffe',
        'darby-ministerial-otter',
        'darcee-semantic-krill',
        'dniren-mental-galliform',
        'dody-legislative-emu',
        'dorelia-positive-firefly',
        'dorolice-useful-worm',
        'edyth-embarrassing-guan',
        'ellen-surprising-limpet',
        'eydie-southern-ostrich',
        'faun-quick-pony',
        'felice-horrible-sloth',
        'gayel-female-tapir',
        'glynda-melodic-sturgeon',
        'guenna-eldest-turkey',
        'gussi-standard-flea',
        'helaine-big-eel',
        'jany-outer-deer',
        'jillian-horrible-herring',
        'joanne-impressive-koala',
        'joey-breakable-dove',
        'jordanna-mild-fowl',
        'jorrie-parallel-angelfish',
        'junie-selective-peafowl',
        'kania-amazing-dragonfly',
        'kathe-whole-catshark',
        'kellia-obedient-falcon',
        'kirby-decisive-cattle',
        'leola-horizontal-mastodon',
        'lissie-improved-roadrunner',
        'lonni-sure-wildfowl',
        'lorena-linguistic-condor',
        'luce-wet-bat',
        'lyda-functional-grasshopper',
        'malissia-chosen-alligator',
        'marianne-continental-wallaby',
        'maureene-universal-platypus',
        'melisande-good-beetle',
        'mella-unconscious-falcon',
        'modestine-asleep-penguin',
        'mora-repulsive-ant',
        'murial-fit-asp',
        'myrna-brainy-gecko',
        'myrta-scattered-chimpanzee',
        'nadya-electronic-ferret',
        'nalani-popular-wren',
        'robinett-positive-newt',
        'ronna-democratic-anteater',
        'rosemarie-ugly-flamingo',
        'ruby-naughty-cow',
        'sabine-different-hamster',
        'saloma-jolly-chicken',
        'salome-evolutionary-mollusk',
        'shari-essential-kingfisher',
        'shellie-formal-rodent',
        'sibel-elegant-stork',
        'susette-shaggy-shrew',
        'tersina-selfish-chameleon',
        'tessa-specified-manatee',
        'thomasa-local-catshark',
        'tilda-female-ladybug',
        'val-shaky-nightingale',
        'van-dead-butterfly',
        'vevay-incredible-asp',
        'wendi-capable-cockroach',
        'willetta-flat-lemming',
        'yovonnda-dutch-bobolink',
        'annemarie-reluctant-clownfish',
        'arda-representative-wasp',
        'ardith-internal-cobra',
        'benedetta-multiple-cattle',
        'bryana-functional-guineafowl',
        'camille-fresh-dragon',
        'charmine-cautious-firefly',
        'constantina-variable-monkey',
        'daria-forthcoming-canidae',
        'darsie-adorable-lynx',
        'doro-used-cheetah',
        'editha-thoughtless-eagle',
        'eran-extended-unicorn',
        'evania-fair-sloth',
        'flory-adequate-tortoise',
        'gayleen-long-monkey',
        'gennifer-wandering-giraffe',
        'georgia-mute-sturgeon',
        'georgia-silky-cattle',
        'germana-comparable-ladybug',
        'golda-eldest-porcupine',
        'goldarina-disturbing-damselfly',
        'guenevere-official-booby',
        'isadora-busy-nightingale',
        'koren-unusual-rodent',
        'kyrstin-imaginative-guppy',
        'laverne-overseas-gazelle',
        'lizbeth-grieving-mockingbird',
        'milli-disturbing-bear',
        'mufinella-likely-piranha',
        'nerte-grumpy-takin',
        'pamela-formidable-turtle',
        'pauline-corresponding-porcupine',
        'perle-cuddly-angelfish',
        'phelia-other-tiglon',
        'rebecca-sure-crane',
        'rikki-specific-clownfish',
        'sallee-average-mole',
        'salomi-roasted-basilisk',
        'selie-bored-ocelot',
        'shelba-pleased-dove',
        'stacie-exceptional-pony',
        'tatiana-difficult-gibbon',
    ]
    
    out_dict = defaultdict(list)
    files = list(zip(scan_files, labels_files))
    with mp.Pool(32) as p:
        i = p.imap_unordered(process_case, files)
        t = tqdm(i, total=len(scan_files))
        for label_file, result in t:
            for k, v in result.items():
                if not v:
                    continue

                out_dict[k].append(label_file.name[:-7])    

    for k, v in out_dict.items():
        with open(f'{k}.txt', 'w') as f:
            for stem in sorted(v):
                f.write(f'{stem}\n')
