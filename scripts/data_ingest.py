import argparse
import json
from pathlib import Path

from eegdash import EEGDash


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple command line argument parser")

    # Add arguments
    parser.add_argument('--data', type=str, default="/mnt/nemar/openneuro/ds004186", help="Path to data directory (Default: /mnt/nemar/openneuro/ds004186)")
    parser.add_argument('--dataset', type=str, default="ds004186", help="Dataset name (Default: ds004186)")

    # Parse the arguments
    args = parser.parse_args()
    print('Arguments:', args)

    obj = EEGDash(
        is_public=False,
    )
    hbn_datasets = ['ds005507','ds005506', 'ds005510', 'ds005512','ds005505','ds005508','ds005509','ds005514','ds005511', 'ds005515', 'ds005516']
    hed_datasets = ['ds004853','ds004852','ds004851','ds004850','ds004849','ds004844','ds004843','ds004842','ds004841','ds004661','ds004660','ds004657','ds004362','ds004123','ds004122','ds004121','ds004120','ds004119','ds004118','ds004117','ds004106','ds004105','ds003645','ds003061','ds002893','ds002691','ds002680','ds002578']
    eeglab_datasets = ["ds004362", "ds005514", "ds002181","ds004554", "ds005697", "ds004151", "ds003800", "ds004350","ds004105", "ds004785", "ds004504", "ds004122", "ds004118","ds004121", "ds004635", "ds005787", "ds005512", "ds005079","ds004120", "ds004119", "ds005178", "ds004019", "ds005342","ds004745", "ds004502", "ds005505", "ds005034", "ds004563","ds002680", "ds003774", "ds004123", "ds003805", "ds005506","ds003838", "ds005507", "ds004040", "ds005511", "ds002718","ds002691", "ds003690", "ds003061", "ds005672", "ds003775","ds004106", "ds005410", "ds005508", "ds005510", "ds005509","ds002578", "ds003620"]

    config_path = Path(__file__).parent / 'datasets.json'
    with open(config_path, 'r') as f:
        datasets_config = json.load(f)
    failed_ds = set(datasets_config['failed_datasets'])

    datasets = ["ds004841","ds004770","ds004561","ds005261","ds000247","ds005131","ds003753","ds003420","ds005028","ds005557","ds005170","ds004840","ds004855","ds004718","ds002725","ds005565","ds004408","ds004796","ds002550","ds004511","ds002893","ds003682","ds004817","ds000248","ds003190","ds004819","ds005089","ds003822","ds003670","ds005048","ds004917","ds004574","ds004852","ds004357","ds003082","ds005574","ds005397","ds004519","ds004602","ds004784","ds005491","ds003846","ds002799","ds004024","ds005815","ds003694","ds005429","ds004771","ds003518","ds004977","ds003702","ds004577","ds005207","ds005866","ds004127","ds003574","ds004703","ds005779","ds004398","ds003523","ds005558","ds004212","ds004347","ds005185","ds005489","ds005398","ds004588","ds001787","ds003505","ds005670","ds003568","ds003703","ds005811","ds004370","ds005340","ds003987","ds004865","ds005363","ds005121","ds004078","ds003392","ds004317","ds004851","ds004033","ds004011","ds003876","ds004166","ds005691","ds005087","ds004330","ds004256","ds004315","ds005279","ds005420","ds003474","ds002034","ds003509","ds004186","ds003825","ds005868","ds003516","ds004587","ds005415","ds004942","ds004348","ds003633","ds004598","ds005383","ds003195","ds004473","ds005403","ds002908","ds004621","ds005863","ds003848","ds004625","ds005594","ds002336","ds004043","ds003517","ds005083","ds004368","ds004584","ds004012","ds003374","ds005624","ds005810","ds003506","ds005106","ds004284","ds005620","ds004738","ds004849","ds005234","ds003570","ds003490","ds002720","ds005307","ds002094","ds002833","ds002218","ds000117","ds004117","ds005021","ds004194","ds005356","ds004264","ds004446","ds004980","ds002722","ds004457","ds004505","ds004853","ds002885","ds004580","ds003944","ds005545","ds004279","ds005876","ds004532","ds004346","ds003816","ds005385","ds004572","ds005095","ds004696","ds004460","ds004902","ds005189","ds005274","ds004075","ds004447","ds004295","ds003519","ds004107","ds004952","ds003458","ds002724","ds003004","ds005571","ds003104","ds004200","ds002791","ds004015","ds005592","ds004262","ds004850","ds005273","ds002712","ds004520","ds004444","ds004582","ds002723","ds004017","ds004595","ds004626","ds003751","ds004475","ds000246","ds004515","ds003421","ds002158","ds004951","ds005522","ds004883","ds004483","ds005065","ds004624","ds004802","ds004993","ds004278","ds004816","ds003739","ds005873","ds004389","ds003194","ds004356","ds004367","ds004369","ds004381","ds004196","ds005692","ds002338","ds004022","ds004579","ds004859","ds005416","ds004603","ds004752","ds003768","ds003947","ds004229","ds005530","ds004844","ds005555","ds004998","ds004843","ds004477","ds001785","ds005688","ds003766","ds004276","ds005540","ds004152","ds004944","ds001971","ds003352","ds003626","ds002814","ds003645","ds005007","ds004551","ds005586","ds001784","ds004809","ds003922","ds004388","ds003810","ds004306","ds004642","ds003478","ds004100","ds003969","ds004000","ds005411","ds004842","ds005305","ds005494","ds004995","ds005114","ds004854","ds003638","ds004521","ds002761","ds001849","ds003844","ds003039","ds004706","ds004252","ds004448","ds005795","ds003602","ds005169","ds003380","ds004018","ds004080","ds004324","ds003887","ds004789","ds004860","ds004837","ds005241","ds003688","ds005107","ds002721","ds003655","ds004395","ds004147","ds003483","ds003555","ds005486","ds005520","ds005262","ds002778","ds004661","ds003885","ds004657","ds005523","ds003498","ds003522","ds005406","ds003710","ds003343","ds003708","ds002001","ds005345","ds004067","ds003078","ds003801","ds005059","ds003029","ds001810","ds005296","ds004660"]
    datasets = hbn_datasets
    for i, ds in enumerate(datasets):
        if ds in failed_ds:
            continue
        try:
            if i % 50 == 0:
                print('Saving failed datasets')
                with open(config_path, 'w') as f:
                    json.dump({'failed_datasets': list(failed_ds)}, f)
            print(f'Processing {ds}')
            obj.add_bids_dataset(dataset=ds, data_dir=f'/mnt/nemar/openneuro/{ds}', overwrite=True)
        except Exception as e:
            print(e)
            failed_ds.add(ds)
            pass

    print(f'Failed datasets: {list(failed_ds)}')
    with open(config_path, 'w') as f:
        json.dump({'failed_datasets': list(failed_ds)}, f)

if __name__ == "__main__":
    main()
