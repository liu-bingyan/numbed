import os
import numpy as np
import pandas as pd
from collections import defaultdict
from utils.io_utils_flat import output_dir_flat, ResultFileNameSturct, get_output_path_flat
from utils.stat_tests import McNemarTest, WilcoxonTest, objective_list



if __name__ == "__main__":
    directory = os.path.join(output_dir_flat, "predictions")
    files = [file for file in os.listdir(directory) if file.endswith('.npy')]
    
    init_comparison = defaultdict(lambda : defaultdict(dict))
    for file in files:
        result = ResultFileNameSturct.from_filename(file)
        config, init = result.config_init()
        init_comparison[config][result.extension][init] = file
    
    report = []
    for config, config_data in init_comparison.items():
        tester = McNemarTest()
        for extension, extension_data in config_data.items():
            uniform_filename = extension_data['uniform']
            alpha_filename = extension_data['alpha']
            uniform_data = np.load(os.path.join(directory,uniform_filename))
            alpha_data = np.load(os.path.join(directory,alpha_filename))
            mcnemar_chi2, mcnemar_p, (a,b,c,d) = tester.eval(uniform_data, alpha_data)
            report.append({'config': config, 'fold': extension, 
                            'a': a, 'b': b, 'c': c, 'd': d,
                            'mcnemar_chi2': mcnemar_chi2, 'mcnemar_p': mcnemar_p,
                        })
    report = pd.DataFrame(report)
    report.to_csv(os.path.join(directory, "mcnemar_test_results.csv"), index=False)