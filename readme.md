# MBUpgrade: Badge Placement Script

Author: Renee St Louis (rstlouis@wpi.edu)  
Last Modified: October 2021

1. [MBUpgrade: Badge Placement Script](#mbupgrade-badge-placement-script)
   1. [Overview](#overview)
   2. [Setup](#setup)
      1. [Installation](#installation)
      2. [Modification](#modification)
      3. [Input Files](#input-files)
   3. [Recommended Usage](#recommended-usage)
   4. [All Flags](#all-flags)
   5. [Uses](#uses)
      1. [List Badges](#list-badges)
      2. [Prepare Data](#prepare-data)
      3. [Recommendations](#recommendations)
      4. [Assignments](#assignments)
   6. [Advanced Usage](#advanced-usage)

## Overview
MBUpgrade is meant to be a multi-purpose script to help manage badge assignments
for MBU. The script exhibits different behaviors based on the flags and
arguments passed to the script.

There are essentially four functions the script can perform:
1. [List Badges](#list-badges)
2. [Prepare Data](#prepare-data)
3. [Recommendations](#recommendations)

4. [Assignments](#assignments)

To see flags, run the following command: `python3 badge_placment.py --help`

## Setup
### Installation
This script requires a python3 environment with the following packages
installed:
- click
- pandas
- numpy

### Modification
There is one part of the script that may require modification, and that is the
preference column names array. It must contain the exact titles of the columns
holding the preferences in the preferences spreadsheet. A sample array is shown
below:

```py
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
```

Additionally, the `name_str` should be updated if the name of the column
containing scout names does not match the string `"Name"`.

### Input Files
The script requires three input files:
- An input csv for preferences, which can have any name
- An input csv for which badges are offered for which period. This file must be
  in a subdirectory, `./source_data/badges.csv`.
- A dictionary of aliases `./typo_dicts.py`

Sample badge csv (`./source_data/badges.csv`):
```csv
p1,p1 capacity,p2,p2 capacity,p3,p3 capacity
Astronomy,0,Chemistry,20,Art,20
Chess,20,Chess,0,Cit Community,20
Cit Community,40,Cit Nation,40,Cit World,40
Cit Nation,40,Space Ex,20,Digital Technology,20
Electronics,20,Cit World,40,Energy,40
E Prep,40,Communications,20,Engineering,20
E Sci,40,Electricity,20,E Sci,20
Music,20,Engineering,20,First Aid,40
Programming,20,First Aid,40,Programming,20
Soil/Water Cons,0,Photography,20,American Business,0
```

Sample Alias Dictionary (`./typo_dicts.py`)
```py
badge_typo_dict = {
 "Chem": "Chemistry",
 "Cit in Community": "Cit Community",
 "Cit nation": "Cit Nation",
 "Citizenship in the Community": "Cit Community",
 "Em Prep": "E Prep",
 "Photo": "Photography",
 "Progreamming": "Programming",
 "Sales": "American Business",
 "Soil/Water": "Soil/Water Cons",
}

```

## Recommended Usage
The recommended use of script to assign badges comes in three steps: cleaning
the data, checking recommendations, and making assignments. For information on
cleaning the data, see the [list badges](#list-badges) section. For information
on checking recommendations, see the [recommendations](#recommendations)
section. For information on making assignments, see the
[assignments](#assignments) section.

Running the script 

The recommended commands for each mode are summarized as follows:
- List Badges: `python3 badge_placement.py -l PREF_FILE`
- Recommendations: `python3 badge_placement.py -r PREF_FILE`
- Make Assignments: `python3 badge_placment.py -ait 50 PREF_FILE`

For a list of the script's flags, see below, or run the following command:

`python3 badge_placment.py --help`

## All Flags
```txt
Usage: badge_placement.py [OPTIONS] PREF_FILE
```

```txt
Options:
  -l, --list-badges               List all badges scouts have selected, then
                                  exit.

  -p, --prepare-data              Clean badge signups using typo_dict, then
                                  exit.

  -b, --stop-before-clear         If cleaning, stops before clear. Otherwise
                                  No op.

  -i, --interest-after-clean      Gauges scout interest after cleaning
                                  entries, not before.

  -a, --archive                   Archive all CSVs from this run.
  -r, --recommendations           Recommend how many sections to have for each
                                  badge, then exit.

  -t, --tournament-rounds INTEGER
                                  If present, the number of assignments to
                                  compare.

  -s, --seed INTEGER              Pass in an int to be used as a random seed.
  --help                          Show this message and exit.
```
## Uses 

### List Badges
MBU form data is notoriously messy. This script requires badge names to be typed
consistently with an exact spelling, meaning the data will need some cleaning.
The easiest way to clean this data is by listing all unique strings entered in
the preference spreadsheet, and comparing them to the list of badges in the
badges csv file. This is exactly what the script does.

Before starting, ensure that all occurrences of the badge names are consistent
in the badges csv. Then run the script with the following configuration:  

`python3 badge_placement.py -l PREF_FILE`  

This will print a series of four lists to console and to a file,
`./out/badge_status.txt`. This list has all badges in the badge csv, all badges
the scouts requested, and the set differences between these in both directions.
The most important group is the badges that scouts requested that are not
offered. These entries must either be replaced or discarded for the script to
work properly.

The script's mechanism for replacing these erroneous badges is with the
`./typo_dicts.py` file. This file contains a dictionary that can be used to
alias different badge names. When the script is run with the above
configuration, it also produces a text file `./out/badge_dict.txt`. This file
contains a list of all erroneous badges formatted into a Python dictionary where
all right-side values are empty strings.

The user can enter the correct name of each badge on the right side of each
entry. An example of a formatted dictionary is shown [above](#input-files).
Entries that are indecipherable or that otherwise cannot be mapped to any badge
should be discarded (deleted from the dictionary). Once the dictionary contains
all badges that will be aliased, it should be moved to the `./typo_dicts.py`
file. 

If the `./typo_dicts.py` file exists and is not empty, the substitutions it
contains may not have been included in the `./out/badge_dict.txt` file. Thus, it
may be advisable to incorporate the new entries into the existing entries if the
existing entries seem agreeable.

Once this is done, you can rerun the above command (`python3 badge_placement.py
-l PREF_FILE`) to make sure that the erroneous badge list seems right (i.e. that
no entry not accounted for can be mapped reliably to an existing badge). When
all badges seem in order, move to the next step.

### Prepare Data
This step is not strictly necessary, but it may be useful. The script can
prepare the data without running assignments. This will produce a file,
`./out/clean.csv`, which will show what the cleaned preference spreadsheet will
look like when the script is operating on it. It may be nice for peace of mind
to compare this to the original spreadsheet.

To run this step, use the following command: 

`python3 badge_placment.py -p PREF_FILE`

### Recommendations
This step can be invaluable for seeing which sections would be advantageous to
duplicate for a given period and where best to spend instructor resources. This
step requires data that has been [cleaned](#list-badges).

To produce a list of recommendations, run the following command:

`python3 badge_placement.py -r PREF_FILE`

This will produce three files:
- `./out/rec/Period_1_Rec.csv`
- `./out/rec/Period_2_Rec.csv`
- `./out/rec/Period_3_Rec.csv`

Each file is a table where each row is a badge, and each column contains an
estimated number of scouts who may want to take the badge. The first column
shows how many slots would be needed to accommodate every scout getting their
first choice of badge. The second column shows how many slots would be needed to
accommodate every scout getting their first or second choice of badge. This is
estimated by counting a scout's first and second preference as half a scout
being in each badge. The third column extends this by putting a third of a scout
in every badge a scout expresses interest in.

Notably, this does not properly account for missing data. For example, a scout
who only puts a first preference without a second or third should be counted as
a full scout in each column. However, this is not done.

You may use these numbers to play around with different class sizes by adjusting
the sizes allowed in the `badges.csv` file.

### Assignments
The script's final usage is its ultimate goal: assigning scouts to badges. This
is the script's default behavior, requiring no flags. However, there are a few
flags that are recommended for best results:

- `-a --archive`: The archive flag saves the run in its own folder in
  `./out/archive/`. 
  - You can leave this flag unset if you are doing many runs and do not want to
    be overwhelmed with many files and folders to keep track of, but this flag
    can be helpful for searching many assignments for preferable configurations.
- `-i --interest-after-clean`: This flag essentially treats scouts who only put
  invalid badges for a period as if they did not express interest at all. 
  - Thus, the script makes no attempt to assign them any badges for this period.

  - You may leave this flag unset if you want to attempt to assign these scouts
    anyway. However, this may hinder performance with particularly messy data,
    which MBU data often is.
  - If this flag is set, the list of un-interested scouts will need to be
    manually re-examined later, but it almost certainly will need to be anyway.
- `-t --tournament`: The best option. This allows the user to specify an integer
  number of iterations to run at once. The script will produce the best
  assignment pattern found among these.

The recommended command is as follows:

`python3 badge_placement.py -ait 50 PREF_FILE`

This will run a tournament of 50 rounds with the above specifications. This will
produce the following files:
- `./out/placements.csv`
- `./out/receipts/UTC_TIME-SCORE.md`
- `./out/ids/`
  - `Period_1_Badges_ID.csv`
  - `Period_2_Badges_ID.csv`
  - `Period_3_Badges_ID.csv`
- `./out/id_match/`
  - `Period_1_Badges.csv`
  - `Period_2_Badges.csv`
  - `Period_3_Badges.csv`
- `./out/sorted/`
  - `Period_1_Badges_Sorted.csv`
  - `Period_2_Badges_Sorted.csv`
  - `Period_3_Badges_Sorted.csv`
- `./out/archive/UTC_TIME-SCORE/`
  - `Period_1_Badges_ID.csv`
  - `Period_2_Badges_ID.csv`
  - `Period_3_Badges_ID.csv`
  - `Period_1_Badges.csv`
  - `Period_2_Badges.csv`
  - `Period_3_Badges.csv`
  - `Period_1_Badges_Sorted.csv`
  - `Period_2_Badges_Sorted.csv`
  - `Period_3_Badges_Sorted.csv`
  - `placements.csv`

The __receipt__ file contains many statistics for the run, which can be used to
check on the run's quality. One helpful check might be to make sure that the
number of scouts accounted for (in the _Total_ column of _Preference
Statistics_) does not change. 

The __placements__ file is the cleaned preference csv with three new columns
listing the badges selected for each scout. There is also a fourth new column
with each scout's script-assigned _id_. These should be checked against the
other spreadsheets.

The other files list all scouts in each badge for one period. These files come
in three flavors. 
1. The first lists each scout by ID. This shows how the spreadsheet sorted each
   scout. This may be helpful for scouts with a blank name, which will not show
   up in other flavors. 
2. The second flavor aliases each scout ID with the scout's name. This can be
   used in combination with the first flavor to improve readability.
3. The third flavor sorts these aliases for maximum human readability. Any data
   for scouts without a name in the `Name` column is essentially lost in this
   view.

The receipts and archived files are named so that they will be easily searchable
and will not overwrite each other. The first part of the name is the time in
milliseconds (since Jan 01 1970). This ensures that the bottommost file will
always be the most recent (assuming alphabetic sorting). The second part of the
name is the best score achieved in the run. This allows the user to find the run
with the best score among many without requiring the opening of any files.

## Advanced Usage
The most powerful feature of this script is that it allows the user to
experiment with how useful it would be for placement to open or close different
sections of different badges. To experiment with this, the user can change the
class sizes in the badges csv to reflect multiple sections of a course or the
removal of a badge (i.e. capacity 0).

If a run is favorable or buggy, and the user wants to reproduce it for any
reason, this should be possible by running the same command to execute the
script with one additional argument: `-s --seed`. Passing in an integer seed can
recreate the random number generation used in a run. A run's seed can be found
in its receipt listed as the __Prime Seed__.