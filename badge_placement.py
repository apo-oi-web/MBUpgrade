import click
from os.path import exists
from os import mkdir
import pandas as pd
import typo_dicts
from itertools import chain
import numpy as np
from sys import version_info, version
from random import randrange
from datetime import datetime

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
    """
    Extracts all preferences from the pref file for the given period. Returns
    result as a data frame.
    :param pref: The scout badge preference data frame
    :param period: The period to extract
    :return: new data frame containing preferences for the given period
    """
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
    """
    Produces a list of preferences for each period
    :param pref: The scout badge preference data frame
    :return: A tuple of three data frames, each containing a period of
    preferences
    """
    return extract_period(pref, 1), extract_period(pref, 2), extract_period(pref, 3)


def list_courses(periods, badges, quiet=False):
    """
    Complies a list of the courses offered and the courses scouts have signed up
    for, along with the difference between those lists in both directions.
    These lists are printed to a log file, out/badge_status.txt. The erroneous
    badges are also printed to a file, out/badge_dict.txt, which the user can
    edit and copy into typo_dicts.py to help clean data.
    :param periods: A tuple of three data frames, each containing a period of
    preferences
    :param badges: A data frame containing the badges offered
    :param quiet: A flag of whether or not to print output to console
    :return: A dictionary with a key for every erroneous entry and every value
    is np.nan.
    """
    courses_registered = set()

    with open('out/badge_status.txt', 'w' if exists('out/badge_status.txt') else 'x') as fp:
        # Small function to write and print
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
            if course and type(course) is str:
                wp('- ' + str(course))

        courses_offered = {""}
        for index, row in badges.iterrows():
            courses_offered.add(row['p1'])
            courses_offered.add(row['p2'])
            courses_offered.add(row['p3'])
        wp("\nCourses we're offering\n----------------------")
        for course in cs(courses_offered):
            if course and type(course) is str:
                wp('- ' + str(course))

        wp("\nEmpty courses\n-------------")
        for course in courses_offered:
            if course and (course not in cs(courses_registered)) and type(course) is str:
                wp('- ' + str(course))

        error_dict = {"": np.nan}
        with op('out/badge_dict.txt') as dict_fp:
            wp("\nErroneous Courses\n-----------------")
            dict_fp.write('badge_typo_dict = {\n')
            for course in cs(courses_registered):
                if course and (course not in courses_offered) and type(course) is str:
                    wp('- ' + str(course))
                    dict_fp.write(f' "{course}": "",\n')
                    error_dict[course] = np.nan
            dict_fp.write('}\n')

    return error_dict


def clean_typos(pref, badges, clean_errors=False):
    """
    Modifies the pref data frame by performing substitutions as prescribed by
    the typo_dicts.py file. If clean_errors is set,
    :param pref: The scout badge preference data frame to modify
    :param badges: The data frame of badges offered each period
    :param clean_errors: A flag of whether or not to remove badges that do not
    match the offered badges after the substitutions are made. This is set by
    default (as is recommended), with the --stop-before-clear flag disabling it.
    """
    pref.replace(typo_dicts.badge_typo_dict, inplace=True)
    error_dict = list_courses(extract_periods(pref), badges, quiet=clean_errors)
    if(clean_errors):
        pref.replace(error_dict, inplace=True)
        list_courses(extract_periods(pref), badges)
        shift_badges_left(pref, badges)
    pref.to_csv('out/clean.csv', index=False)


def shift_badges_left(pref, badges):
    """
    Modifies preference data frame by shifting entries to the left if there are
    any gaps. This function also attempts to remove duplicates. This function
    has a few known bugs, such as leaving both gaps and duplicate functions if
    all three preferences for a period and scout match. This should be mostly
    inconsequential thanks to a few safties in the assignment algorithm.
    :param pref: The scout badge preference data frame
    :param badges: The data frame of badges offered each period
    """
    col_names = list(chain.from_iterable(pref_col_names))
    for period in pref_col_names:
        for i in [0, 1]:
            to_replace = pref.loc[:, period[i]].isnull()
            pref.loc[to_replace, period[i]] = pref.loc[to_replace, period[i + 1]]
            pref.loc[to_replace, period[i+1]] = np.nan


def make_recommendations(periods, badges):
    """
    Tallies how many scouts have signed up for each badge for each period, then
    prints these results to three files in out/rec/.
    :param periods: An array of three data frames, each containing a period of
    preferences
    :param badges: The data frame of badges offered in each period
    """
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
    """
    Returns a data frame with n rows and 3 columns to describe the preferences
    of n scouts across 3 periods. Each entry will be True if a scout is
    interested in taking classes for the period, as shown by listing at least
    one preference. All entries with no preferences listed will be false.
    :param periods: A collection of three data frames, each containing all scout
    preferences for one period
    :return: A boolean data frame where entry [r,c] is True if scout r listed at
    least one preference in period c.
    """
    def is_interested(row):
        return (row['first'] is not np.nan) or (row['second'] is not np.nan) or (row['third'] is not np.nan)

    interested = pd.DataFrame(np.ones((len(periods[0].index), 3), dtype=bool))
    for i, period in enumerate(periods):
        interested.iloc[:, i] = period.apply(is_interested, axis=1)
    return interested


def get_part_time_scouts(periods):
    """
    Returns a boolean series of part-time scouts. A scout is part-time if
    they have listed any preferences for at least one period.
    :param periods: A collection of three data frames, each containing all scout
    preferences for one period
    :return: A boolean Series where entry [n] is True if scout n listed no
    preferences for at least one period
    """
    p1_chosen = periods[0].loc[:, 'first'].isnull()
    p2_chosen = periods[1].loc[:, 'first'].isnull()
    p3_chosen = periods[2].loc[:, 'first'].isnull()
    return p1_chosen | p2_chosen | p3_chosen


def get_picky_scouts(periods):
    """
    Returns a boolean series of picky scouts. A scout is picky if they have
    listed fewer than three preferences for a given period. This is not a tight
    or robust definition, but it should not create problems for the assignment
    algorithm.
    :param periods: A collection of three data frames, each containing all scout
    preferences for one period
    :return: A boolean Series where entry [n] is True if scout n listed fewer
    than three preferences for any period
    """
    p1_chosen_1 = periods[0].loc[:, 'second'].isnull()
    p1_chosen_2 = periods[0].loc[:, 'third'].isnull()
    p2_chosen_1 = periods[1].loc[:, 'second'].isnull()
    p2_chosen_2 = periods[1].loc[:, 'third'].isnull()
    p3_chosen_1 = periods[2].loc[:, 'second'].isnull()
    p3_chosen_2 = periods[2].loc[:, 'third'].isnull()
    return p1_chosen_1 | p1_chosen_2 | p2_chosen_1 | p2_chosen_2 | p3_chosen_1 | p3_chosen_2


def get_missed_scouts(unassigned_df):
    """
    Returns a boolean series of scouts that are currently unassigned. A scout is
    unassigned if they have not been placed into a badge for at least one
    period.
    :param unassigned_df: A boolean data frame of all scouts who have not yet
    been assigned to a badge. Entry [r,c] is True if scout r has not been placed
    into a badge for period c.
    :return: A boolean series of scouts that are currently unassigned. A scout
    is unassigned if they have not been placed into a badge for at least one
    period.
    """
    return unassigned_df.iloc[:, 0] | unassigned_df.iloc[:, 1] | unassigned_df.iloc[:, 2]


def assign_scouts_by_id(periods, assignments, unassigned_df, ranks, seed, ids=None, use_all_prefs=False):
    """
    Attempts to assign scouts with the given ids to badges in every period. If
    no ids are provided, the default is to use ids of all scouts who have not
    yet been assigned to badges.
    :param periods: A collection of three data frames, each containing all scout
    preferences for one period
    :param assignments: A collection of three dictionaries, where each key is a
    badge in the corresponding period, and the value is a tuple where the first
    item is the badge capacity and the second item is a set of ids for scouts
    signed up to take it.
    :param unassigned_df: A boolean data frame of all scouts who have not yet
    been assigned to a badge. Entry [r,c] is True if scout r has not been placed
    into a badge for period c.
    :param ranks: An array where each entry represents the placements for a
    period. This representation is a sub-array of three entries, one to count
    scouts who got their first choice, one for second, and one for other.
    :param seed: The random seed that should be used to shuffle the data frame
    of chosen scouts
    :param ids: A series of ids of scouts to attempt to assign.
    :param use_all_prefs: A flag for whether or not to attempt to use scout
    preferences for all periods to attempt to assign the current period. This
    can help remedy erroneous data entry and can better utilize class spaces.
    """
    def is_room(tup):
        return len(tup[1]) < tup[0]

    def has_pref(r):
        return (r['first'] is not np.nan) or (r['second'] is not np.nan) or (r['third'] is not np.nan)

    def assign(badge):
        """
        Ineternal function to attempt to assign the active scout to the given
        badge. Removes that badge from the scout's preferences on success so
        that the scout is not assigned to the same badge more than once.
        :param badge: A string representing the badge to attempt to assign the
        scout to.
        :return: True if the assignment was successful, false otherwise.
        """
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
                        if periods[j].loc[row['id'], k] == badge:
                            periods[j].loc[row['id'], k] = np.nan
                return True
        return False

    # Randomize which period goes first
    for i in pd.Series([0, 1, 2]).sample(frac=1, random_state=seed):
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
                    """
                    Internal function that uses all of a scout's listed
                    preferences to attempt to assign them to a badge for the
                    active period. This function will only be called if the
                    use_all_prefs flag is true and if badge assignment using
                    preferences for this period has failed.
                    :return: None. This function returns to break out of nested
                    loops to multiple assignments of the same scout.
                    """
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
    """
    Attempts to assign scouts into given badges using preferences to badges in
    the badges data frame. Attempts different assignments, starting with part-
    time scouts (see get_part_time_scouts()), then picky scouts (see get_picky_
    scouts()), then all scouts, then all scouts using all preferences.
    :param pref: The data frame of scout badge preferences
    :param badges: The data frame of badges offered in each period
    :param interested: A boolean data frame of interested scouts. Entry [r,c] is
    True if scout r listed at least one preference in period c.
    :param seed: The random seed to be used for shuffling the data frame of
    chosen scouts.
    :return: The data structure of assignments, a boolean data frame of
    unassigned scouts, and the list of ranks for each period. See
    assign_scouts_by_id documentation for details.
    """
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
    assign_scouts_by_id(periods, assignments, unassigned_df, ranks, seed, use_all_prefs=True)
    return assignments, unassigned_df, ranks


def conduct_tournament(pref, badges, interested, best, tournament_rounds, time, archive=False):
    """
    Conducts a tournament by repeatedly attempting to assign scouts using
    different procedurally-generated random seeds. Each assignment is scored,
    and the assignment with the best score is kept. Returns the best
    configuration and an array of all scores.
    :param pref: The data frame of all scout badge preferences
    :param badges: The data frame of badges offered in each period
    :param interested: A boolean data frame of interested scouts. Entry [r,c] is
    True if scout r listed at least one preference in period c.
    :param best: A dictionary of statistics about the best configuration found
    so far.
    :param tournament_rounds: An int representing the number of rounds to iterate
    through.
    :param time: The program's start time, used for logging.
    :param archive: A flag of whether or not to archive the results to a folder
    in out/archive/.
    :return: The statistics of the best assignment found, and an array of all
    tournamnet scores.
    """
    np.random.seed(best['seed'])  # Generate predictable seeds based on prime seed
    scores = np.array([best['score']])
    print('\nStarting Tournament\n===================\n')
    for r in range(tournament_rounds - 1):
        if ((r + 2) % 10) is 0:
            print(f'- Tournament Round {r + 2:3d} Top Score {scores.max()}')
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
    export_periods(pref, badges, best['assignments'], best['unassigned'], interested, archive, time, best['score'])
    print(f'\nTournament Results\n------------------')
    print(f'score: {best["score"]}')
    print(f'mean score: {scores.mean()}')
    print(f'median score: {np.median(scores)}')
    print(f'low score: {scores.min()}')
    print(f'winning seed: {best["seed"]}')
    return best, scores


def export_periods(pref, badges, assignments, unassigned_df, interested, archive=False, time=None, score=None):
    """
    Exports the given assignments to many CSVs. One CSV is placements.csv, which
    adds placement data to the original preference file. The others are files
    that show all badge assignments for a period, and these files are grouped in
    threes, one for each period. There are three flavors of these files: ids,
    names, and alphabetized names. These files will be exported into different
    folders in out/. If the archive flag is set, these files will also be
    exported to a folder in out/archive/.
    :param pref: The data frame of all scout badge preferences
    :param badges: The data frame of badges offered in each period
    :param assignments: A collection of three dictionaries, where each key is a
    badge in the corresponding period, and the value is a tuple where the first
    item is the badge capacity and the second item is a set of ids for scouts
    signed up to take it.
    :param unassigned_df: A boolean data frame of all scouts who have not yet
    been assigned to a badge. Entry [r,c] is True if scout r has not been placed
    into a badge for period c.
    :param interested: A boolean data frame of interested scouts. Entry [r,c] is
    True if scout r listed at least one preference in period c.
    :param archive: A flag of whether or not to archive the results to a folder
    in out/archive/.
    :param time: The program's start time, used for logging and file names to
    keep logs distinct and chronologically sorted.
    :param score: The score of the assignments being exported.
    """
    def get_size(p, s):
        return np.array(s).sum()

    def get_ids(s):
        return list(np.array(s).nonzero()[0])

    def pad_set(s, length):
        s = list(s)
        for _ in range(int(length) - len(s)):
            s.append(np.nan)
        return s

    time = int(time.timestamp())
    if not exists('out/ids'):
        mkdir('out/ids')
    if not exists('out/sorted'):
        mkdir('out/sorted')
    if not exists('out/id_match'):
        mkdir('out/id_match')
    for i in range(3):
        results = {}
        not_placed = unassigned_df.iloc[:, i] & interested.iloc[:, i]
        max_size = max(badges[f'p{i+1} capacity'].max(), get_size(i, not_placed), get_size(i, ~interested.iloc[:, i]))
        for badge, data in sorted(assignments[i].items()):
            results[badge] = pad_set(data[1], max_size)
        results['Unassigned'] = pad_set(get_ids(not_placed), max_size)
        results['Not Interested'] = pad_set(get_ids(~interested.iloc[:, i]), max_size)  # Bitwise Not

        results = pd.DataFrame.from_dict(results)
        results_id = results.copy()
        results.to_csv(f'out/ids/Period_{i+1}_Badges_ID.csv', index=False)
        name_dict = dict(zip(pref[id_str], pref[name_str]))
        results.replace(name_dict, inplace=True)
        results_matched = results.copy()
        results.to_csv(f'out/id_match/Period_{i+1}_Badges.csv', index=False)

        # Sort CSV and pull some shenanigans to sort np.nan
        zst = 'zzzzzzzz'
        results.replace(np.nan, zst, inplace=True)
        for col in results.columns:
            results.loc[:, col] = sorted(list(results.loc[:, col]))
        results.replace(zst, np.nan, inplace=True)

        results.to_csv(f'out/sorted/Period_{i + 1}_Badges_Sorted.csv', index=False)

        if archive and (time is not None):
            if not exists('out/archive'):
                mkdir('out/archive')
            if not exists(f'out/archive/{time}-{score}'):
                mkdir(f'out/archive/{time}-{score}')
            results_id.to_csv(f'out/archive/{time}-{score}/Period_{i+1}_Badges_ID.csv', index=False)
            results_matched.to_csv(f'out/archive/{time}-{score}/Period_{i+1}_Badges.csv', index=False)
            results.to_csv(f'out/archive/{time}-{score}/Period_{i+1}_Badges_Sorted.csv', index=False)

    # Combine with primary spreadsheet
    pref_placed = pref.copy()
    placements = pd.DataFrame(np.full((len(pref.index), 3), np.nan))
    placements[not_placed] = 'Unassigned'  # order matters
    placements[~interested] = 'Not Interested'
    print('\nPlacement Warnings\n==================\n')
    print('If the end of a line says anything other than "Unassigned", be concerned.')
    for p, p_dict in enumerate(assignments):
        for badge, roster in p_dict.items():
            for scout in roster[1]:
                if type(scout) is not int:  # filter empty badges
                    continue
                if type(placements.iloc[scout, p]) is str:
                    print(f'- Scout {name_dict[scout]:25} with id {scout:3d} was placed into badge {badge:20} but had status {placements.iloc[scout, p]:20}')
                placements.iloc[scout, p] = badge
    print('')
    pos = pref_placed.columns.get_loc(pref_col_names[0][0])
    pref_placed.insert(loc=pos, column='First Period Badge', value=placements.iloc[:, 0])
    pref_placed.insert(loc=pos+1, column='Second Period Badge', value=placements.iloc[:, 1])
    pref_placed.insert(loc=pos+2, column='Third Period Badge', value=placements.iloc[:, 2])
    pref_placed.to_csv(f'out/placements.csv', index=False)
    if archive and (time is not None):
        pref_placed.to_csv(f'out/archive/{time}-{score}/placements.csv', index=False)


def score_assignments(unassigned_df, ranks, interested):
    """
    Produces a numeric representation for how preferable an assignment is.
    Higher preferences are rewarded, and failing to assign scouts who were
    interested in being assigned is penalized.
    :param unassigned_df: A boolean data frame of all scouts who have not yet been
    assigned to a badge. Entry [r,c] is True if scout r has not been placed into
    a badge for period c.
    :param ranks: An array where each entry represents the placements for a
    period. This representation is a sub-array of three entries, one to count
    scouts who got their first choice, one for second, and one for other.
    :param interested: A boolean data frame of interested scouts. Entry [r,c] is
    True if scout r listed at least one preference in period c.
    :return: An int representing how preferable an assignment is.
    """
    score = 0
    for successes in ranks:
        score += 3 * successes[0]
        score += 2 * successes[1]
        score += 1 * successes[2]
    score -= 10 * np.sum(np.array(unassigned_df))
    score += 10 * np.sum(np.array(1 - interested))
    return score


def print_receipt(badges, best, scores, ranks, seed, iac, time, interested, archive=False, quiet=False):
    """
    Prints a log of this run to console and to a file in out/receipts/. This
    receipt includes details on the best score for this run, how many
    preferences were respected, how many scouts were not assigned, and so on.
    :param badges: The data frame of badges offered in each period
    :param best: The statistics of the best assignment found this run.
    :param scores: An array of all scores achieved during a tournament, or an
    empty list if there was no tournament.
    :param ranks: An array where each entry represents the placements for a
    period. This representation is a sub-array of three entries, one to count
    scouts who got their first choice, one for second, and one for other.
    :param seed: The prime seed used to generate all other seeds this run.
    Entering the prime seed with the -s flag should in theory allow a run to be
    duplicated deterministically.
    :param iac: A copied value from the --interest-after-clear flag. Must be
    logged for any chance of reproducability.
    :param time: The program's start time, used for logging and file names to
    keep logs distinct and chronologically sorted.
    :param interested: A boolean data frame of interested scouts. Entry [r,c] is
    True if scout r listed at least one preference in period c.
    :param archive: A flag of whether or not to archive the results to a folder
    in out/archive/.
    :param quiet: A flag for whether or not to suppress printing to console.
    """
    fp_archive = None
    tst = int(time.timestamp())
    file_name = f'{tst}-{best["score"]}'

    def wp(s, suppress_print=False):
        fp.write(f'{s}\n')
        if not quiet and not suppress_print:
            print(s)
        if fp_archive is not None:
            fp_archive.write(f'{s}\n')

    def do_the_thing():
        """
        Internal function to help with the conditional existence of file
        pointers. Produces a log file with the given settings from the outer
        function.
        """
        if not exists('out/receipts'):
            mkdir('out/receipts')
        wp(f'MBU Sorting Script Receipt\n==========================\n')
        wp(f'Start Time: {time}  ')
        wp(f'End Time: {datetime.now()}  ')
        wp(f'File: {file_name}  ')
        wp(f'\nFinal Score: __{best["score"]}__  ')
        wp(f'Prime Seed: {seed}  ')
        wp(f'Interest After Clean: {iac}  ')
        wp(f'\n## Preference Statistics\n')
        wp(f'| Period | First Pref | Second Pref | Third+ Pref | Sum | Unassigned | Not Interested | Total |')
        wp(f'| :----- | ---------: | ----------: | ----------: | --: | ---------: | -------------: | ----: |')
        for i, p in enumerate(ranks):
            not_interested = np.sum(np.array(1 - interested)[:, i])
            not_assigned = np.sum(np.array(best['unassigned'] & interested)[:, i])
            s = p[0] + p[1] + p[2] + not_interested + not_assigned
            wp(f'| {i+1:6} | {p[0]:10d} | {p[1]:11d} | {p[2]:11d} | {p[0]+p[1]+p[2]:3d} | {not_assigned:10d} | {not_interested:14d} | {s:5d} |')
        wp('\nNote: Numbers may be inconsistent with aliased and sorted spreadsheets if there are blank names.\n')
        if len(scores) > 1:
            wp(f'\n## Tournament Results\n')
            for i in range(9, len(scores), 10):
                wp(f'- Tournament Round {i+1:3d} Top Score {scores[0:i].max()}')
            wp(f'\n## Tournament Statistics\n')
            wp(f'- Tournament Rounds: {len(scores)}')
            wp(f'- Top Score: {scores.max()}')
            wp(f'- Mean Score: {scores.mean()}')
            wp(f'- Median Score: {np.median(scores)}')
            wp(f'- Lowest Score: {scores.min()}')
            wp(f'- Winning Seed: {best["seed"]}')
        wp(f'\n## Class Size Inputs\n')
        wp(f'| Period 1             | Size | Period 2             | Size | Period 3             | Size |')
        wp(f'| :------------------- | ---: | :------------------- | ---: | :------------------- | ---: |')
        for index, row in badges.iterrows():
            wp(f'| {row["p1"]:20} | {row["p1 capacity"]:4d} | {row["p2"]:20} | {row["p2 capacity"]:4d} | {row["p3"]:20} | {row["p3 capacity"]:4d} |')
        wp('')  # New Line
        wp('```txt', suppress_print=True)
        wp(f'▀██    ██▀ ▀██▀▀█▄   ▀██▀  ▀█▀  \n'
          + ' ███  ███   ██   ██   ██    █  \n'
          + ' █▀█▄▄▀██   ██▀▀▀█▄   ██    █  \n'
          + ' █ ▀█▀ ██   ██    ██  ██    █  \n'
          + '▄█▄ ▀ ▄██▄ ▄██▄▄▄█▀    ▀█▄▄▀')
        wp('```', suppress_print=True)
    if archive:
        with op(f'out/receipts/{file_name}.md') as fp:
            with op(f'out/archive/{file_name}/{file_name}.md') as fp_archive:
                do_the_thing()
    else:
        with op(f'out/receipts/{file_name}.md') as fp:
            do_the_thing()


@click.command()
@click.option('--list-badges', '-l', is_flag=True, help='List all badges scouts have selected, then exit.')
@click.option('--prepare-data', '-p', is_flag=True, help='Clean badge signups using typo_dict, then exit.')
@click.option('--stop-before-clear', '-b', is_flag=True, help='If cleaning, stops before clear. Otherwise No op.')
@click.option('--interest-after-clean', '-i', is_flag=True, help='Gauges scout interest after cleaning entries, not before.')
@click.option('--archive', '-a', is_flag=True, help="Archive all CSVs from this run.")
@click.option('--recommendations', '-r', is_flag=True, help='Recommend how many sections to have for each badge, then exit.')
@click.option('--tournament-rounds', '-t', required=False, help="If present, the number of assignments to compare.", type=int)
@click.option('--seed', '-s', required=False, help='Pass in an int to be used as a random seed.', type=int)
@click.argument('pref-file', type=click.Path(exists=True))
def main(pref_file, list_badges=None, prepare_data=None, stop_before_clear=None, interest_after_clean=None, archive=None, recommendations=None, tournament_rounds=None, seed=None):
    """
    Main function to execute this program. All parameters are flags set using
    the click package. Flags and arguments are described thus:
    :param pref_file: An argument for the csv file containing all scout badge
    preferences.
    :param list_badges: A flag that, if set, causes the program to list all
    badges offered in the badge CSV, requested in the preference CSV, and the
    set differences between them. This does not use the alias dictionary from
    the external file.
    :param prepare_data: A flag that, if set, causes the program to terminate
    after the preference CSV has been cleaned. The resuling file can be found at
    out/clean.csv.
    :param stop_before_clear: A flag that, if set, prevents the preference CSV
    from being cleaned of values not found in the badges offered.
    :param interest_after_clean: A flag that, if set, delays gauging scout
    interest until after the preference CSV has been cleaned. The recommended
    setting for this flag is true. This helps ignore data for badges not
    offered.
    :param archive: A flag that, if set, saves assignments from this run in an
    archive folder. This prevents the assignment from getting overwritten by
    another run.
    :param recommendations: A flag that, if set, causes the program to produce
    recommendations in the out/rec/ folder, then terminate. These
    recommendations show how many souts signed up for each badge, and they can
    be used to help decide which badges may warrant additional sections.
    :param tournament_rounds: An option for how many rounds to hold in the
    tournament of competing assignments. If not set, only one run is conducted.
    :param seed: An option for a prime seed to use for this run. Must be an int
    between 0 and 2^32 - 1
    """
    time = datetime.now()
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1, dtype=int)
    seed = int(seed)
    # np.random.seed(seed)

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
    best = {
        'score': score,
        'assignments': assignments,
        'unassigned': unassigned_df,
        'ranks': ranks,
        'seed': seed,
    }
    if tournament_rounds is None:
        export_periods(pref, badges, assignments, unassigned_df, interested, archive, time, score)
        print(f'score: {score}')
        print_receipt(badges, best, [], ranks, seed, interest_after_clean, time, interested)
    else:

        best, scores = conduct_tournament(pref, badges, interested, best, tournament_rounds, time, archive)
        print_receipt(badges, best, scores, ranks, seed, interest_after_clean, time, interested)


if __name__ == '__main__':
    main()
