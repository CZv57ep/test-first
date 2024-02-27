"""
Query FTC investigations data and display some basic statistics.

"""

import numpy as np

import mergeron.core.ftc_merger_investigations_data as fid

invdata = fid.construct_data(fid.INVDATA_ARCHIVE_PATH)

for data_period in invdata:
    print(data_period, "-->")
    for table_type in (isd1 := invdata[data_period]):
        leader_str = "\t"
        print(leader_str, table_type, "-->")
        leader_str += "\t"
        for table_no in (isd11 := isd1[table_type]):
            (invdata_ind_group, invdata_evid_cond, table_data_array) = isd11[table_no]
            print(
                leader_str,
                table_no,
                " \N{EM DASH} ",
                invdata_ind_group,
                f", {invdata_evid_cond or 'N/A'}",
                ", ",
                sep="",
                end="",
            )
            print(
                "Odds ratio = {}/{}".format(
                    *np.einsum("ij->j", table_data_array[:, -3:])
                )
            )
    print("\n")

data_period, data_type, table_no = ("2004-2011", "HHI and Delta", "Table 3.3")
#  data_period, data_type, table_no = "2004-2011", "Firm Count", "Table 4.1"
print(f"Investigations data, {data_period}, by {data_type}, {table_no}")
print(
    "{}, {}\n{}".format(
        *invdata[data_period][f"By{data_type.replace(' ', '')}"][table_no]
    )
)
