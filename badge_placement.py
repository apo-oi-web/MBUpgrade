import click
from os.path import exists
from os import mkdir
import pandas as pd
import typo_dicts
from itertools import chain
import numpy as np
from sys import version_info, version
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
        wp("Badges Offered and Resuested\n============================\n")
        wp("Courses that scouts signed up for\n---------------------------------")
        for course in cs(courses_registered):
            if course and course is not np.nan:
                wp('- ' + str(course))

        courses_offered = {""}
        for index, row in badges.iterrows():
            courses_offered.add(row['p1'])
            courses_offered.add(row['p2'])
            courses_offered.add(row['p3'])
        wp("\nCourses we're offering\n----------------------")
        for course in cs(courses_offered):
            if course and course is not np.nan:
                wp('- ' + str(course))

        wp("\nEmpty courses\n-------------")
        for course in courses_offered:
            if course and (course not in cs(courses_registered)) and course is not np.nan:
                wp('- ' + str(course))

        error_dict = {"": np.nan}
        with op('out/badge_dict.txt') as dict_fp:
            wp("\nErroneous Courses\n-----------------")
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


def make_recommendations(periods, badges):
    if not exists('out/rec'):
        mkdir('out/rec')
    for i, period in enumerate(periods):
        period_rec = {}

        for badge in badges[['p1', 'p2', 'p3'][i]]:
            period_rec[badge] = [0, 0, 0]

        for index, row in period.iterrows():
            # How many sections would be needed to get everyone into their first choice?
            badge = row['first']
            if badge in period_rec.keys():
                period_rec[badge][0] += 1
                period_rec[badge][1] += 1/2
                period_rec[badge][2] += 1/3

            # How many sections would be needed to get everyone into their second choice?
            badge = row['second']
            if badge in period_rec.keys():
                period_rec[badge][1] += 1/2
                period_rec[badge][2] += 1/3

            # How many sections would be needed to get everyone into their third choice?
            badge = row['third']
            if badge in period_rec.keys():
                period_rec[badge][2] += 1/3

        recommendations = pd.DataFrame.from_dict(period_rec, orient='index', columns=['First Preference', 'First or Second', 'Any Preference'])
        recommendations.to_csv(f'out/rec/Period_{i+1}_Rec.csv')


def get_interested_scouts(periods):
    def is_interested(row):
        return (row['first'] is not np.nan) or (row['second'] is not np.nan) or (row['third'] is not np.nan)

    interested = pd.DataFrame(np.ones((len(periods[0].index), 3), dtype=bool))
    for i, period in enumerate(periods):
        interested.iloc[:, i] = period.apply(is_interested, axis=1)
    return interested


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
    return unassigned_df.iloc[:, 0] | unassigned_df.iloc[:, 1] | unassigned_df.iloc[:, 2]


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

    # Randomize which period goes first
    for i in pd.Series([0, 1, 2]).sample(frac=1):
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
            # Careful not to roll this into a loop. _Continue_ is necessary for outer loop.
            if assign(row['first']):
                ranks[i][0] += 1
                continue
            if assign(row['second']):
                ranks[i][1] += 1
                continue
            if assign(row['third']):
                ranks[i][2] += 1
                continue

            # Attempt to use preferences listed for other badges
            if use_all_prefs:
                # Define function to break out of nested loops
                def go_ham():
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
                            return
                go_ham()


def assign_scouts(pref, badges, interested, seed):
    periods = extract_periods(pref)
    assignments = [{}, {}, {}]
    unassigned_df = interested.copy()
    # unassigned_df = pd.DataFrame(np.ones((len(pref.index), 3), dtype=bool))
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
    # assign_scouts_by_id(periods, assignments, unassigned_df, ranks, seed, get_missed_scouts(unassigned_df), use_all_prefs=True)
    assign_scouts_by_id(periods, assignments, unassigned_df, ranks, seed)
    # assign_scouts_by_id(periods, assignments, unassigned_df, unassigned_df, ranks, seed, use_all_prefs=True)
    return assignments, unassigned_df, ranks


def conduct_tournament(pref, badges, interested, best, tournament_rounds):
    np.random.seed(best['seed'])  # Generate predictable seeds
    scores = np.array([best['score']])
    print('\nStarting Tournament\n===================\n')
    for r in range(tournament_rounds - 1):
        if ((r + 2) % 10) is 0:
            print(f'- Tournament Round {r + 2} Top Score {scores.max()}')
        round_seed = np.random.randint(0, 2 ** 32 - 1, dtype=int)
        assignments, unassigned_df, ranks = assign_scouts(pref, badges, interested, round_seed)
        score = score_assignments(unassigned_df, ranks, interested)
        scores = np.append(scores, score)
        if score > best['score']:
            best = {
                'score': score,
                'assignments': assignments,
                'unassigned': unassigned_df,
                'ranks': ranks,
                'seed': round_seed,
            }
    export_periods(pref, badges, best['assignments'], best['unassigned'], interested)
    print(f'\nTournament Results\n------------------')
    print(f'score: {best["score"]}')
    print(f'mean score: {scores.mean()}')
    print(f'median score: {np.median(scores)}')
    print(f'low score: {scores.min()}')
    print(f'winning seed: {best["seed"]}')


def export_periods(pref, badges, assignments, unassigned_df, interested):
    def get_size(p, s):
        return np.array(s).sum()

    def get_ids(s):
        return list(np.array(s).nonzero()[0])

    def pad_set(s, length):
        s = list(s)
        for _ in range(int(length) - len(s)):
            s.append(np.nan)
        return s

    if not exists('out/ids'):
        mkdir('out/ids')
    if not exists('out/sorted'):
        mkdir('out/sorted')
    if not exists('out/id_match'):
        mkdir('out/id_match')
    for i in range(3):
        results = {}
        not_placed = unassigned_df.iloc[:, i] & interested.iloc[:, i]
        max_size = max(badges[f'p{i+1} capacity'].max(), get_size(i, not_placed), get_size(i, interested))
        for badge, data in sorted(assignments[i].items()):
            results[badge] = pad_set(data[1], max_size)
        results['Unassigned'] = pad_set(get_ids(not_placed), max_size)
        results['Not Interested'] = pad_set(get_ids(~interested.iloc[:, i]), max_size)  # Bitwise Not

        results = pd.DataFrame.from_dict(results)
        results.to_csv(f'out/ids/Period_{i+1}_Badges_ID.csv', index=False)
        results.replace(dict(zip(pref[id_str], pref[name_str])), inplace=True)
        results.to_csv(f'out/id_match/Period_{i+1}_Badges.csv', index=False)

        # Sort CSV and pull some shenanigans to sort np.nan
        zst = 'zzzzzzzz'
        results.replace(np.nan, zst, inplace=True)
        for col in results.columns:
            results.loc[:, col] = sorted(list(results.loc[:, col]))
        results.replace(zst, np.nan, inplace=True)

        results.to_csv(f'out/sorted/Period_{i + 1}_Badges_Sorted.csv', index=True)


def score_assignments(unassigned_df, ranks, interested):
    score = 0
    for successes in ranks:
        score += 3 * successes[0]
        score += 2 * successes[1]
        score += 1 * successes[2]
    score -= 10 * np.sum(np.array(unassigned_df))
    score += 10 * np.sum(np.array(1 - interested))
    return score


@click.command()
@click.option('--list-badges', '-l', is_flag=True, help='List all badges scouts have selected, then exit.')
@click.option('--prepare-data', '-p', is_flag=True, help='Clean badge signups using typo_dict, then exit.')
@click.option('--stop-before-clear', '-b', is_flag=True, help='If cleaning, stops before clear. Otherwise No op.')
@click.option('--interest-after-clean', '-i', is_flag=True, help='Gauges scout interest after cleaning entries, not before.')
@click.option('--recommendations', '-r', is_flag=True, help='Recommend how many sections to have for each badge, then exit.')
@click.option('--tournament-rounds', '-t', required=False, help="If present, the number of assignments to compare.", type=int)
@click.option('--seed', '-s', required=False, help='Pass in an int to be used as a random seed.', type=int)
@click.argument('pref-file', type=click.Path(exists=True))
def main(pref_file, list_badges=None, prepare_data=None, stop_before_clear=None, interest_after_clean=None, recommendations=None, tournament_rounds=None, seed=None):

    if seed is None:
        seed = np.random.randint(0, 2**32 - 1, dtype=int)
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

    interested = get_interested_scouts(extract_periods(pref))
    clean_typos(pref, badges, clean_errors=not stop_before_clear)

    if prepare_data:
        return

    if recommendations:
        make_recommendations(extract_periods(pref), badges)
        return

    if interest_after_clean:
        interested = get_interested_scouts(extract_periods(pref))

    assignments, unassigned_df, ranks = assign_scouts(pref, badges, interested, seed)
    score = score_assignments(unassigned_df, ranks, interested)

    if tournament_rounds is None:
        export_periods(pref, badges, assignments, unassigned_df, interested)
        print(f'score: {score}')
    else:
        best = {
            'score': score,
            'assignments': assignments,
            'unassigned': unassigned_df,
            'ranks': ranks,
            'seed': seed,
        }
        conduct_tournament(pref, badges, interested, best, tournament_rounds)


if __name__ == '__main__':
    main()
