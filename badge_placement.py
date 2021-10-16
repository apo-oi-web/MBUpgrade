import click
from os.path import exists
from os import mkdir
import pandas as pd
import typo_dicts
from itertools import chain
import numpy as np
from sys import version_info, version
from datetime import datetime
from random import randrange

name_str = 'Name'
# id_str = 'Reference #'
id_str = 'id'  # new uuid
pref_col_names = [
    [
        'Please select your FIRST choice for the first (9-11) class',
        'Please select your SECOND choice for the first (9-11) class',
        'Please select your THIRD choice for the first (9-11) class'
    ], [
        'Please select your FIRST choice for the second (12-2) class',
        'Please select your SECOND choice for the second (12-2) class',
        'Please select your THIRD choice for the second (12-2) class',
    ], [
        'Please select your FIRST choice for the third (2:30-4:30) class',
        'Please select your SECOND choice for the third (2:30-4:30) class',
        'Please select your THIRD choice for the third (2:30-4:30) class',
    ]
]


def cs(l):
    return sorted(str(x) for x in l)


def op(filename):
    return open(filename, 'w' if exists(filename) else 'x')


def extract_period(pref, period):

    period_pref = pref[[
        id_str,
        name_str,
        pref_col_names[period - 1][0],
        pref_col_names[period - 1][1],
        pref_col_names[period - 1][2]
    ]].copy()
    period_pref.rename(columns = {
        pref_col_names[period - 1][0]: 'first',
        pref_col_names[period - 1][1]: 'second',
        pref_col_names[period - 1][2]: 'third',
    }, inplace=True)
    return period_pref


def extract_periods(pref):
    return extract_period(pref, 1), extract_period(pref, 2), extract_period(pref, 3)


def list_courses(periods, badges, quiet=False):
    """
    Complies a list of the courses offered and the courses scouts
    have signed up for, along with the difference between those
    lists in both directions.
    """
    courses_registered = set()

    with open('out/badge_status.txt', 'w' if exists('out/badge_status.txt') else 'x') as fp:
        def wp(s):
            fp.write(s + '\n')
            if not quiet:
                print(s)

        for period in periods:
            for index, row in period.iterrows():
                courses_registered.add(row['first'])
                courses_registered.add(row['second'])
                courses_registered.add(row['third'])
        if not quiet:
            print('')  # New line for my sanity
        wp("Courses that scouts signed up for\n=================================")
        for course in cs(courses_registered):
            if course and course is not np.nan:
                wp('- ' + str(course))

        courses_offered = {""}
        for index, row in badges.iterrows():
            courses_offered.add(row['p1'])
            courses_offered.add(row['p2'])
            courses_offered.add(row['p3'])
        wp("\nCourses we're offering\n======================")
        for course in cs(courses_offered):
            if course and course is not np.nan:
                wp('- ' + str(course))

        wp("\nEmpty courses\n=============")
        for course in courses_offered:
            if course and (course not in cs(courses_registered)) and course is not np.nan:
                wp('- ' + str(course))

        error_dict = {"": np.nan}
        with op('out/badge_dict.txt') as dict_fp:
            wp("\nErroneous Courses\n=================")
            dict_fp.write('badge_typo_dict = {\n')
            for course in cs(courses_registered):
                if course and (course not in courses_offered) and course is not np.nan:
                    wp('- ' + str(course))
                    dict_fp.write(f' "{course}": "",\n')
                    error_dict[course] = np.nan
            dict_fp.write('}\n')

    return error_dict


def clean_typos(pref, badges, clean_errors=False):
    pref.replace(typo_dicts.badge_typo_dict, inplace=True)
    error_dict = list_courses(extract_periods(pref), badges, quiet=clean_errors)
    if(clean_errors):
        pref.replace(error_dict, inplace=True)
        list_courses(extract_periods(pref), badges)
        shift_badges_left(pref, badges)
    pref.to_csv('out/clean.csv', index=False)


def shift_badges_left(pref, badges):
    col_names = list(chain.from_iterable(pref_col_names))
    for period in pref_col_names:
        for i in [0, 1]:
            to_replace = pref.loc[:, period[i]].isnull()
            pref.loc[to_replace, period[i]] = pref.loc[to_replace, period[i + 1]]
            pref.loc[to_replace, period[i+1]] = np.nan


def get_part_time_scouts(periods):
    p1_chosen = periods[0].loc[:, 'first'].isnull()
    p2_chosen = periods[1].loc[:, 'first'].isnull()
    p3_chosen = periods[2].loc[:, 'first'].isnull()
    return p1_chosen | p2_chosen | p3_chosen


def get_picky_scouts(periods):
    p1_chosen_1 = periods[0].loc[:, 'second'].isnull()
    p1_chosen_2 = periods[0].loc[:, 'third'].isnull()
    p2_chosen_1 = periods[1].loc[:, 'second'].isnull()
    p2_chosen_2 = periods[1].loc[:, 'third'].isnull()
    p3_chosen_1 = periods[2].loc[:, 'second'].isnull()
    p3_chosen_2 = periods[2].loc[:, 'third'].isnull()
    return p1_chosen_1 | p1_chosen_2 | p2_chosen_1 | p2_chosen_2 | p3_chosen_1 | p3_chosen_2

def get_missed_scouts(unassigned_df):
    return unassigned_df[:, 0] | unassigned_df[:, 1] | unassigned_df[:, 2]


def assign_scouts_by_id(periods, assignments, unassigned_df, ranks, seed, ids=None, use_all_prefs=False):
    def is_room(tup):
        return len(tup[1]) < tup[0]

    def has_pref(r):
        return (r['first'] is not np.nan) or (r['second'] is not np.nan) or (r['third'] is not np.nan)

    def assign(badge):
        # If the badge is offered this period
        if badge in assignments[i].keys():
            # If there is a preference and room
            if (badge is not np.nan) and is_room(assignments[i][badge]):
                badge_roster = assignments[i][badge][1]
                badge_roster.add(row['id'])
                unassigned_df.iloc[row['id'], i] = False

                # Remove the preference so that scouts won't be assigned to duplicate classes across periods
                for j in range(3):
                    for k in ['first', 'second', 'third']:
                        if periods[j].loc[row['id'], k] is badge:
                            periods[j].loc[row['id'], k] = np.nan

                return True
        return False

    for i in range(len(periods)):
        # If there are ids to prioritize
        if ids is not None:
            records_to_select = ids & unassigned_df.iloc[:, i]
        else:
            records_to_select = periods[i].apply(has_pref, axis=1)
            records_to_select &= unassigned_df[i]
        chosen = periods[i][records_to_select]  # Careful to make a copy
        chosen = chosen.sample(frac=1, random_state=seed)  # shuffle
        for index, row in chosen.iterrows():

            # If the scout has already been assigned, stop
            if not unassigned_df.iloc[row['id'], i]:
                continue
            for p_num, p_name in enumerate(['first', 'second', 'third']):
                if assign(row[p_name]):
                    ranks[i][p_num] += 1
                    continue

            # Attempt to use preferences listed for other badges
            if use_all_prefs:
                all_prefs = []

                # Get other badges
                for other_period in [(i + 1) % 3, (i + 2) % 3]:
                    for other_p_name in ['first', 'second', 'third']:
                        other_badge = periods[other_period].loc[row['id'], other_p_name]
                        if other_badge is not np.nan:
                            all_prefs.append(other_badge)
                for badge in all_prefs:
                    if assign(badge):
                        # Hard-coded as third-preference equivalent
                        ranks[i][1] += 1
                        continue


def assign_scouts(pref, badges, seed):
    periods = extract_periods(pref)
    assignments = [{}, {}, {}]
    unassigned_df = pd.DataFrame(np.ones((len(pref.index), 3), dtype=bool))
    ranks = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(len(periods)):
        for badge in badges[f'p{i+1}']:
            if badge is np.nan:
                continue
            capacity = badges.loc[badges[f'p{i+1}'] == badge, f'p{i+1} capacity']
            # Make an empty list for every badge in the period
            assignments[i][badge] = (int(capacity.iloc[0]), set())

    assign_scouts_by_id(periods, assignments, unassigned_df, ranks, seed, ids=get_part_time_scouts(periods))
    assign_scouts_by_id(periods, assignments, unassigned_df, ranks, seed, ids=get_picky_scouts(periods))
    # assign_scouts_by_id(periods, assignments, unassigned_df, ranks, seed, get_missed_scouts(periods, unassigned), use_all_prefs=True)
    assign_scouts_by_id(periods, assignments, unassigned_df, ranks, seed)
    # assign_scouts_by_id(periods, assignments, unassigned, unassigned_df, ranks, seed, use_all_prefs=True)
    return assignments, unassigned_df, ranks


def pad_set(s, length):
    s = list(s)
    for _ in range(int(length) - len(s)):
        s.append(np.nan)
    return s


def export_periods(pref, badges, assignments, unassigned_df):
    for i in range(3):
        results = {}
        max_size = max(badges[f'p{i+1} capacity'].max(), np.array(unassigned_df.iloc[:, i]).sum())
        for badge, data in sorted(assignments[i].items()):
            results[badge] = pad_set(data[1], max_size)
        results['Unassigned'] = list(np.array(unassigned_df.iloc[:, i]).nonzero()[0])

        results = pd.DataFrame.from_dict(results)
        results.to_csv(f'out/Period_{i+1}_Badges_ID.csv', index=False)
        results.replace(dict(zip(pref[id_str], pref[name_str])), inplace=True)
        results.to_csv(f'out/Period_{i+1}_Badges.csv', index=False)

        zst = 'zzzzzzzz'
        results.replace(np.nan, zst, inplace=True)
        for col in results.columns:
            results.loc[:, col] = sorted(list(results.loc[:, col]))
        results.replace(zst, np.nan, inplace=True)
        results.to_csv(f'out/Period_{i + 1}_Badges_Sorted.csv', index=True)


def score_assignments(unassigned_df, ranks):
    score = 0
    for successes in ranks:
        score += 3 * successes[0]
        score += 2 * successes[1]
        score += 1 * successes[2]
    score -= 10 * np.sum(np.array(unassigned_df))
    return score


@click.command()
@click.option('--list-badges', '-l', is_flag=True, required=False, help='List all badges scouts have selected, then exit.')
@click.option('--prepare-data', '-p', is_flag=True, required=False, help='Clean badge signups using typo_dict, then exit.')
@click.option('--stop-before-clear', '-b', is_flag=True, required=False, help='If cleaning, stops before clear. Otherwise No op.')
@click.option('--seed', '-s', required=False, help='Pass in an int to be used as a random seed.')
@click.argument('pref-file', type=click.Path(exists=True))
def main(list_badges=None, prepare_data=None, stop_before_clear=None, seed=None, pref_file=None):

    if seed is None:
        seed = randrange(0, (2**32) - 1)
    seed = int(seed)

    if version_info.major < 3:
        raise Exception (f'This script requires Python 3. You are using version {version}.')

    if exists(pref_file):
        pref = pd.read_csv(pref_file)
    else:
        raise Exception('Please provide a preference CSV as an argument.')
    # fk = Faker()
    # fk.seed(seed)
    # pref['id'] = [fk.uuid4() for _ in range(len(pref.index))]
    pref['id'] = range(len(pref.index))

    if exists('source_data/badges.csv'):
        badges = pd.read_csv('source_data/badges.csv')
    else:
        raise Exception('Please provide a list of badges at source_data/badges.csv')

    if not exists('out'):
        mkdir('out')

    if list_badges:
        list_courses(extract_periods(pref), badges)
        return

    clean_typos(pref, badges, clean_errors=not stop_before_clear)

    if prepare_data:
        return

    assignments, unassigned_df, ranks = assign_scouts(pref, badges, seed)
    export_periods(pref, badges, assignments, unassigned_df)
    print(f'score: {score_assignments(unassigned_df, ranks)}')


if __name__ == '__main__':
    main()
