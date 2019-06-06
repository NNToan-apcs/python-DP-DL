import tensorflow as tf
train, test = tf.keras.datasets.mnist.load_data()
# f=open("Train.txt")
# if f.mode == 'r':
#     contents = f.readline()
#     a= contents.find("features")
#     result = ''
#     i=0
#     while(contents):
#         i+=1
#         print(i)
#         features = contents[29+9:]
#         features = features.replace(' ', ',')
#         result = result + features
#         contents = f.readline()
# f.close()

# f=open("train_feat_file.txt", "w+")
# f.write(result)
# f.close()

f=open("Train.txt")
if f.mode == 'r':
    contents = f.readline()
    
    result = ''
    i=0
    
    # labels = labels.replace(' ', ',')
    while(contents):
        i+=1
        print(i)
        labels = contents[8:27]
        labels= int(labels.find('1')/2)
        result = result+ str(labels) + "\n"
        contents = f.readline()
f.close()

f=open("train_label_file.txt", "w+")
f.write(result)
f.close()