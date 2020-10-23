# import twint
# c = twint.Config()
# c.Search = "kuliah daring"
# c.Since = "2020-04-01"
# c.Until = "2020-08-01"
# c.Store_csv = True
# c.Output = "none"
# c.Limit = 100
# c.Output="testing.csv"

# twint.run.Search(c)

import re
import json
import xlsxwriter

text_file = open('CorpusTweet.txt', 'r')
data = text_file.read().split('</DOC>')
text_file.close

def get_regex_group(document):
    result = re.search(r'<KELAS>(.*)</KELAS>\s\s<TEXT>\s(.*)\s\s</TEXT>',document)
    if result:
        return result
    else:
        return None

alldata = list()
for document in data:
    result = get_regex_group(document)
    print(result.group(2))


# with xlsxwriter.Workbook('test.xlsx') as workbook:
#     worksheet = workbook.add_worksheet()

#     for row_num, data in enumerate(alldata):
#         worksheet.write_row(row_num, 0, data)