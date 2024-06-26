import geopandas
import pandas as pd
import sys
import os
import tools


def count_max_consec_detects(nest_data, date_data):
    """Determine the maximum number of consecutive bird detections"""
    assert date_data.shape[0] == 1, "date_data should be a Pandas DataFrame with one row"
    sorted_dates = pd.Series(date_data.Date[0]).sort_values().reset_index(drop=True)
    sorted_nest_dates = pd.Series(nest_data.Date).sort_values().reset_index(drop=True)
    sorted_dates_dict = {val: key for key, val in sorted_dates.items()}
    sorted_dates_combined_diff = sorted_nest_dates.map(sorted_dates_dict).diff()
    all_consec_detects = []
    consec_detects = 0
    for i in range(1, len(sorted_dates_combined_diff)):
        if sorted_dates_combined_diff[i] == 1 and sorted_dates_combined_diff[i - 1] != 1:
            # New start to consectutive detection set
            consec_detects = 1
            if i + 1 == len(sorted_dates_combined_diff):
                all_consec_detects.append(consec_detects)
        elif sorted_dates_combined_diff[i] == 1 and sorted_dates_combined_diff[i - 1] == 1:
            # Increment existing consecutive detection set
            consec_detects += 1
            if i + 1 == len(sorted_dates_combined_diff):
                all_consec_detects.append(consec_detects)
        elif sorted_dates_combined_diff[i] != 1 and sorted_dates_combined_diff[i - 1] == 1:
            # Store completed consecutive detection set and reset
            all_consec_detects.append(consec_detects)
            consec_detects = 0
        elif sorted_dates_combined_diff[i] != 1 and sorted_dates_combined_diff[i - 1] != 1:
            consec_detects == 0
        else:
            assert False, "Oops, I shouldn't be here"
    if all_consec_detects:
        max_consec_detects = max(all_consec_detects)
    else:
        max_consec_detects = 0

    return max_consec_detects


def process_nests(nest_file, year, site, savedir, min_score=0.3, min_detections=3, min_consec_detects=1):
    """Process nests into a one row per nest table"""
    nests_data = geopandas.read_file(nest_file)
    date_data = nests_data.groupby(['Site', 'Year']).agg({'Date': lambda x: x.unique().tolist()}).reset_index()
    target_inds = nests_data['target_ind'].unique()
    nests = []
    for target_ind in target_inds:
        nest_data = nests_data[(nests_data['target_ind'] == target_ind) & (nests_data['score'] >= min_score)]
        num_consec_detects = count_max_consec_detects(nest_data, date_data)
        if len(nest_data) >= min_detections or num_consec_detects >= min_consec_detects:
            summed_scores = nest_data.groupby(['Site', 'Year', 'target_ind', 'label']).score.agg(['sum', 'count'])
            top_score_data = summed_scores[summed_scores['sum'] == max(summed_scores['sum'])].reset_index()
            nest_info = nest_data.groupby(['Site', 'Year', 'target_ind']).agg({
                'Date': ['min', 'max', 'count'],
                'match_xmin': ['mean'],
                'match_ymin': ['mean'],
                'match_xmax': ['mean'],
                'match_ymax': ['mean']
            }).reset_index()
            xmean = (nest_info['match_xmin']['mean'][0] + nest_info['match_xmax']['mean']) / 2
            ymean = (nest_info['match_ymin']['mean'][0] + nest_info['match_ymax']['mean']) / 2
            bird_match = ",".join([str(x) for x in nest_data["bird_id"]])
            nests.append([
                target_ind, nest_info['Site'][0], nest_info['Year'][0], xmean[0], ymean[0], nest_info['Date']['min'][0],
                nest_info['Date']['max'][0], nest_info['Date']['count'][0], top_score_data['label'][0],
                top_score_data['sum'][0], top_score_data['count'][0], bird_match
            ])

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    filename = os.path.join(savedir, f"{site}_{year}_processed_nests.shp")

    if nests:
        nests = pd.DataFrame(nests,
                             columns=[
                                 'nest_id', 'Site', 'Year', 'xmean', 'ymean', 'first_obs', 'last_obs', 'num_obs',
                                 'species', 'sum_top1', 'num_top1', 'bird_match'
                             ])
        nests_shp = geopandas.GeoDataFrame(nests, geometry=geopandas.points_from_xy(nests.xmean, nests.ymean))
        nests_shp.crs = nests_data.crs
        nests_shp.to_file(filename)
    else:
        schema = {
            "geometry": "Polygon",
            "properties": {
                'nest_id': 'int',
                'Site': 'str',
                'Year': 'str',
                'xmean': 'float',
                'ymean': 'float',
                'first_obs': 'str',
                'last_obs': 'str',
                'num_obs': 'int',
                'species': 'str',
                'sum_top1': 'float',
                'num_top1': 'int',
                'bird_match': 'str'
            }
        }
        crs = nests_data.crs
        empty_nests = geopandas.GeoDataFrame(geometry=[])
        empty_nests.to_file(filename, driver='ESRI Shapefile', schema=schema, crs=crs)


if __name__ == "__main__":
    working_dir = tools.get_working_dir()
    path = sys.argv[1]
    split_path = os.path.normpath(path).split(os.path.sep)
    year = split_path[5]
    site = split_path[6]
    nestdir = os.path.join(working_dir, "processed_nests", year, site)
    process_nests(path, year, site, savedir=nestdir)
