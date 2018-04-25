import sys
file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]

# Tweet text file
f1 = open(file1,'r')

# Tweet label file
labels = []
f2 = open(file2,'r')
for line in f2:
    labels.append(line.strip())

# File to write to!
fw = open(file3,'w')
header = 'Descript\tCategory\n'
fw.write(header)

for i in range(len(labels)):
    line = f1.readline()
    row = line.strip() + '\t' + labels[i] + '\n'
    fw.write(row)

f1.close()
f2.close()
fw.close()
