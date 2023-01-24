with open('./ensemble/result.txt','r') as re:
    pre=re.readlines()

with open('./dataset/source_code.txt','r') as scode:
    code=scode.readlines()

with open('./dataset/comment.txt','r') as scomment:
    comment=scomment.readlines()

for i in range(30):
    print('Source Code: ',code[i].strip())
    print('Original Comment：',comment[i].strip())
    print("Predict Comment",pre[i].strip())
    print("\n")