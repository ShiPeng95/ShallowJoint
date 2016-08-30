# -*- coding: utf-8 -*-


class Word(object):
    def __init__(self,num,word,tag,father,arc,srl):
        self.word = word
        self.tag = tag
        self.father = int(father)
        self.arc = arc
        self.num = int(num)
        self.srl = srl;
class Step(object):
    def __init__(self, stack, queue, action):
        self.stack = stack
        self.queue = queue
        self.action = action

tree_corpus = []
oracle_step = []

def noChild(num, queue):
    for word in queue:
        if word.father==num:
            return False
    return True

def generate(sentence):
    step_list = []
    action_list = []
    stack = []
    queue = sentence + [Word(num=-1, word="ROOT", tag="ROOT-_-ROOT", father=-2, arc="_", srl="ROOT")]
    while (True):
        step_stack = "["
        step_queue = "["
        for item in stack:
            step_stack += item.word+"-"+item.tag+", "
        if len(step_stack)>=2:
            step_stack = step_stack[0:-2]+"]"
        else:
            step_stack += "]"

        for item in queue:
            step_queue += item.word+"-"+item.tag+", "
        if len(step_queue)>=2:
            step_queue = step_queue[0:-2]+"]"
        else:
            step_queue += "]"
        step_list.append(step_stack+step_queue)

        if (len(stack) == 1) and (len(queue) == 0):
            break

        step_action = ""
        if len(stack)>1:
            first = stack[0]
            second = stack[1]
            if second.father == first.num:
                step_action = "LEFT-ARC("+second.arc+")"
                stack.pop(1)
                action_list.append(step_action)
                #print step_action
                continue
            if (first.father == second.num) and noChild(first.num, queue):
                step_action = "RIGHT-ARC("+first.arc+")"
                stack.pop(0)
                action_list.append(step_action)
                #print step_action
                continue
        step_action = "SHIFT"
        #print step_list[-1]
        stack.insert(0,queue.pop(0))
        action_list.append(step_action)
        #print step_action
    return action_list, step_list








def main():
    treefile = open("ChineseSemanticTreebank.txt")

    sentence = []
    i = 0
    j = 0
    for line in treefile.readlines():
        if len(line) == 1:
            tree_corpus.append(sentence)
            #print "Reading Tree ",i," sentence finished"
            sentence = []
            i += 1
            j = 0
            #break
            continue
        wordinfo = line.strip().split()
        word = Word(num=j, word=wordinfo[0], tag=wordinfo[1]+"-"+wordinfo[3]+"-"+wordinfo[4], father=wordinfo[2], arc=wordinfo[3], srl=wordinfo[4])
        sentence.append(word)
        j += 1
    print "reading tree finished"
    '''
    i = 0
    step_list = []
    for line in oraclefile.readlines():
        if len(line)==1:
            oracle_step.append(step_list)
            #print "reading Step ",i," sentence finished"
            step_list = []
            i += 1
            continue
        stepinfo = line.strip().split("][")
        if len(stepinfo) == 1:
            action = stepinfo[0]
            step = Step(stack=stack, queue=queue, action=action)
            step_list.append(step)
        else:
            stack, queue = stepinfo[0][1:], stepinfo[1][0:-1]
    print "reading oracle finished"
        '''

if __name__=="__main__":
    main()
    gene = open("joint.train","w")
    gene.write("\n")

    i = 0
    for sentence in tree_corpus:
        i += 1
        action , step = generate(sentence)
        j = 0
        gene.write(step[0]+"\n")
        for j in range(len(action)):
            gene.write(action[j]+"\n")
            gene.write(step[j+1]+"\n")
        gene.write("\n")
        print "write ",i," to ",gene.name
        if i == 12000:
            gene = open("joint.dev","w")
            gene.write("\n")
        if i == 13000:
            gene = open("joint.test","w")
            gene.write("\n")

