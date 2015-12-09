%%
%%Code for evaluating Exemplar-based Topic Detection
%Author: Ahmed Elbagoury (ahmed.elbagoury@uwaterloo.ca), Rania Ibrahim
%(rania.ibrahim@uwaterloo.ca)
%If you use this paper, please cite the following paper:
%   Elbagoury, Ahmed, et al. "Exemplar-Based Topic Detection in Twitter Streams" Ninth International AAAI Conference on Web and Social Media. 2015.
%Inputs:
%   -dirName: The directory that contains the input tweets. Each time slot
%   is represented by a file, that is in TF-IDF format.
%   -outputDir: The output will be written to this directory. The directory
%    is assumed to contain folder for each run and each run folder contains
%    a folder for each number of required topics.
%       For example if 10 runs are required and the number of required
%       topics are 2 and 4 there should be 10 subfolders (from '1' to '10') and
%       each of these folders contains 2 folders ('2' and '4')
%   -topicsNum: An array containing the number of required topics
%   -random_seed: an array contains the random seed for each of the
%   required number of topics. This ensures reproducibility of the results.
%   -random_set_size: The number of randomly chosend sample used to
%   speed-up the computations (m in the paper)
%   -sim_threshold: similarity threshold (epsilon in the paper)
%Outputs:
%%
function Exemplar_based_Topic_Detection(dirName, outputDir, topicsNum, random_seed, random_set_size, sim_threshold)
    files = dir(dirName);
    for r=1:10
        timefile = fopen(strcat(outputDir,num2str(r),'\time.txt'),'w');
        for t=1:size(topicsNum,2)
            time = 0;
            topicNum = topicsNum(t);
            keywordNum = 15;
            prev_topics = cell(topicNum, 1);
            cur_slot = 0;
            for file = files'
                 if length(file.name) <= 2
                     continue;
                 end
                 cur_slot = cur_slot + 1;
                 file.name;
                 load(strcat(dirName,file.name));
                 output_file = fopen(strcat(outputDir,num2str(r),'\',num2str(topicNum),'\',file.name(1:length(file.name)-4),'.txt'),'w');
                 n = size(mTfIdf,1);
                 m = min(n,random_set_size); % nystrom approximation
                 tic;
                 rng(random_seed(r));
                 rand_perm = randperm(n);
                 mTfIdf_nystrom = mTfIdf(rand_perm', :);
                 k = sparse(mTfIdf_nystrom(1:m,:)*mTfIdf_nystrom');
                 l = diag(sum(mTfIdf_nystrom.*mTfIdf_nystrom, 2).^(-0.5));
                 k = sparse(l(1:m,1:m) * k * l);
                 e = ones(m,1);
                 avg = 1/m * e' * k;
                 c = (k - e * avg);
                 variance = (1/(m-1))*(sum(c.*c,1).^0.5);
                 [sorted_docs, docs_ind] = sort(variance, 'descend');
                 selectedTopics = zeros(topicNum);
                 cur_topic = 1;
                 % force T to one and zero
                 cur_doc = 1;
                 while cur_topic <= topicNum    %Select the cur_topic th topic
                     if cur_topic > 1
                        sim = Inf;
                        while sim > sim_threshold
                             if docs_ind(cur_doc) < m
                                 sim = k(docs_ind(cur_doc), selectedTopics(cur_topic-1));
                             elseif selectedTopics(cur_topic-1) < m
                                 sim = k(selectedTopics(cur_topic-1), docs_ind(cur_doc));
                             else
                                sim = mTfIdf_nystrom(docs_ind(cur_doc),:)*mTfIdf_nystrom(selectedTopics(cur_topic-1),:)';
                                sim = full(sim * (l(docs_ind(cur_doc),docs_ind(cur_doc)) * l (selectedTopics(cur_topic-1), selectedTopics(cur_topic-1))));
                             end

                             if sim <= sim_threshold
                                 break;
                             end
                             cur_doc = cur_doc + 1;
                        end
                     end
                     selectedTopics(cur_topic) = docs_ind(cur_doc);
                     cur_doc = cur_doc + 1;
                     cur_topic = cur_topic + 1;
                 end
                 time = time + toc;
                 prev_topics = cell(topicNum, 1);
                 for i=1:topicNum
                     topicIndex = selectedTopics(i);
                     cur_terms = mTfIdf_nystrom(topicIndex,:);
                     [sorted_terms, term_ind] = sort(cur_terms, 'descend');    %Sort the terms of the top topics  
                     size1 = min(find(sorted_terms == 0));
                     s = min(size1-1, keywordNum);
                     prev_topics{i} = cell(1, s);
                     for j=1:keywordNum 
                        if sorted_terms(j) == 0
                            break;
                        end
                        fprintf(output_file, '%s ', strtrim(cDictionary(term_ind(j),:)));
                        prev_topics{i, j} = strtrim(cDictionary(term_ind(j),:));
                     end
                     fprintf(output_file, '\n');
                 end
                 fclose(output_file);
            end     %End of each time slot
            fprintf(timefile,'%f\n',time);
        end
        fclose(timefile);
    end
end
