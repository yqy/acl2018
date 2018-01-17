#!/usr/bin/env python
#coding=utf-8
import os

def get_all_file(path,dir_list):
    if os.path.isfile(path):
        dir_list.append(path)
    elif os.path.isdir(path):
        for item in os.listdir(path):
            itemsrc = os.path.join(path, item)
            get_all_file(itemsrc,dir_list)
    return dir_list

if __name__ == "__main__":

    raw_dirname = './conll-2012/v9/'
    target_dir = "./conll-new/"

    #Generate new data in $target_dir
    
    files = get_all_file(raw_dirname,[])
    for fil in files:
        if "arabic" in fil:
            continue

        if fil.endswith("auto_conll"):
            dir_file = "/".join(fil.strip().split("/")[1:-1])
            new_dir = target_dir+dir_file
            os.system("mkdir -p %s"%new_dir)

            auto_file = open(fil)

            file_name = fil.strip().split("/")[-1]
            target_file = new_dir+"/"+file_name
            v4_gold_file = fil.replace("v9","v4").replace("auto_conll","gold_conll")
            print >> sys.stderr,"Write",target_file

            gold_file = open(v4_gold_file)
            fw = open(target_file,"w")
            while True:
                line = auto_file.readline()
                if not line:break
                auto_line = line.strip()

                line = gold_file.readline()
                gold_line = line.strip().split(" ")

                new_auto_line = auto_line
                if len(auto_line) > 0 and not auto_line[0] == "#":
                    gold_mention_cluster_info = gold_line[-1]
                    new_auto_line = auto_line[:-1]+gold_mention_cluster_info

                fw.write(new_auto_line.strip()+"\n")
            fw.close()
            os.system("cp %s %s"%(v4_gold_file,target_file.replace("auto_conll","gold_conll")))
