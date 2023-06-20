import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import metrics

from processtransformer import constants
from processtransformer.data import loader
from processtransformer.models import transformer

parser = argparse.ArgumentParser(description="Process Transformer - Next Time Prediction.")

parser.add_argument("--dataset", required=True, type=str, help="dataset name")

parser.add_argument("--model_dir", default="./models", type=str, help="model directory")

parser.add_argument("--result_dir", default="./results", type=str, help="results directory")

parser.add_argument("--next_act_dir", default="./results/next_activity", type=str, help="next_act directory")

parser.add_argument("--next_time_dir", default="./results/next_time", type=str, help="next_time directory")

parser.add_argument("--remaining_time_dir", default="./results/remaining_time", type=str, help="remaining_time directory")

parser.add_argument("--task", type=constants.Task, 
    default=constants.Task.TIMES,  help="task name")

parser.add_argument("--epochs", default=10, type=int, help="number of total epochs")

parser.add_argument("--batch_size", default=64, type=int, help="batch size")

parser.add_argument("--learning_rate", default=0.002, type=float,
                    help="learning rate")

parser.add_argument("--gpu", default=0, type=int, 
                    help="gpu id")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if __name__ == "__main__":
    #创建和保存模型
    model_path = f"{args.model_dir}/{args.dataset}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = f"{model_path}/predict_ckpt"

    result_path = f"{args.result_dir}/{args.dataset}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = f"{result_path}/results"

    next_act_path = f"{args.next_act_dir}"
    if not os.path.exists(next_act_path):
        os.makedirs(next_act_path)
    next_act_path = f"{next_act_path}/next_act_predict"

    next_time_path = f"{args.next_time_dir}"
    if not os.path.exists(next_time_path):
        os.makedirs(next_time_path)
    next_time_path = f"{next_time_path}/next_time_predict"

    remaining_time_path = f"{args.remaining_time_dir}"
    if not os.path.exists(remaining_time_path):
        os.makedirs(remaining_time_path)
    remaining_time_path = f"{remaining_time_path}/remaining_time_predict"

    #创建加载数据对象
    data_loader = loader.LogsDataLoader(name = args.dataset)
    #加载下一个事件数据
    (train_act_df, test_act_df, x_word_dict, y_word_dict, max_case_length,
        vocab_size, num_output) = data_loader.load_data(constants.Task.NEXT_ACTIVITY)
    #加载时间任务数据
    (train_time_df, test_time_df, x_word_dict1, y_word_dict1, max_case_length1,
     vocab_size1, num_output1) = data_loader.load_data(constants.Task.TIMES)

    #准备时间任务数据
    (train_token_x, train_time_x, 
         train_token_next_time, train_token_rmain, time_scaler, y_scaler,y_scaler1) = data_loader.prepare_data_times(train_time_df,
        x_word_dict1, max_case_length1)
    #准备下一个事件任务数据
    train_token_act_x, train_token_act_y = data_loader.prepare_data_next_activity(train_act_df,
                                                                          x_word_dict, y_word_dict, max_case_length)
    #创建模型
    transformer_model = transformer.get_predict_model(
        max_case_length=max_case_length, 
        vocab_size=vocab_size,output_dim=num_output)
    #编译模型
    transformer_model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss={'out1':tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              "out2":tf.keras.losses.LogCosh(),'out3':tf.keras.losses.LogCosh()}
        ,loss_weights = {'out1':0.6,"out2":2,"out3":0.3})


    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_weights_only=True,
        monitor="loss", save_best_only=True)
    #训练模型
    transformer_model.fit([train_token_act_x,train_token_x, train_time_x], [train_token_act_y,train_token_next_time, train_token_rmain],
        epochs=args.epochs, batch_size=args.batch_size, 
        verbose=2, callbacks=[model_checkpoint_callback])


################# check the k-values #########################################
    #评估模型
    k, accuracies,fscores, precisions, recalls, maes, mses, rmses, maes1, mses1, rmses1 = [],[],[],[],[],[],[],[],[],[],[]
    for i in range(max_case_length):
        test_data_subset = test_act_df[test_act_df["k"] == i]
        test_data_subset1 = test_time_df[test_time_df["k"]==i]
        if len(test_data_subset) > 0:
            test_token_act_x, test_y1 = data_loader.prepare_data_next_activity(test_data_subset,
                                                                                x_word_dict, y_word_dict,
                                                                                max_case_length)
            test_token_x, test_time_x, test_y2, test_y3, _, _, _ = data_loader.prepare_data_times(
                test_data_subset1, x_word_dict, max_case_length, time_scaler, y_scaler,y_scaler1, False)

            y_pred = transformer_model.predict([test_token_act_x,test_token_x,test_time_x])
            #下一个事件预测评估
            y_pred1 = np.argmax(y_pred[0],axis=1) #获取分类值
            accuracy = metrics.accuracy_score(test_y1, y_pred1)
            precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                test_y1, y_pred1, average="weighted")
            #下一个事件时间评估
            y_pred2 = y_pred[1]
            _test_y2 = y_scaler.inverse_transform(test_y2)
            _y_pred2 = y_scaler.inverse_transform(y_pred2)
            #剩余时间评估
            y_pred3 = y_pred[2]
            _test_y3 = y_scaler1.inverse_transform(test_y3)
            _y_pred3 = y_scaler1.inverse_transform(y_pred3)

            k.append(i)
            #输出下一个事件结果
            accuracies.append(accuracy)
            fscores.append(fscore)
            precisions.append(precision)
            recalls.append(recall)
            #输出下一个事件时间结果
            maes.append(metrics.mean_absolute_error(_test_y2, _y_pred2))
            mses.append(metrics.mean_squared_error(_test_y2, _y_pred2))
            rmses.append(np.sqrt(metrics.mean_squared_error(_test_y2, _y_pred2)))
            #输出剩余时间事件结果
            maes1.append(metrics.mean_absolute_error(_test_y3, _y_pred3))
            mses1.append(metrics.mean_squared_error(_test_y3, _y_pred3))
            rmses1.append(np.sqrt(metrics.mean_squared_error(_test_y3, _y_pred3)))
    k.append(i + 1)
    accuracies.append(np.mean(accuracy))
    fscores.append(np.mean(fscores))
    precisions.append(np.mean(precisions))
    recalls.append(np.mean(recalls))

    maes.append(np.mean(maes))
    mses.append(np.mean(mses))
    rmses.append(np.mean(rmses))

    maes1.append(np.mean(maes1))
    mses1.append(np.mean(mses1))
    rmses1.append(np.mean(rmses1))

    print('Average accuracy across all prefixes:', np.mean(accuracies))
    print('Average f-score across all prefixes:', np.mean(fscores))
    print('Average precision across all prefixes:', np.mean(precisions))
    print('Average recall across all prefixes:', np.mean(recalls))

    print('Average MAE across all prefixes:', np.mean(maes))
    print('Average MSE across all prefixes:', np.mean(mses))
    print('Average RMSE across all prefixes:', np.mean(rmses))

    print('Average MAE1 across all prefixes:', np.mean(maes1))
    print('Average MSE1 across all prefixes:', np.mean(mses1))
    print('Average RMSE1 across all prefixes:', np.mean(rmses1))
    #写入表格数据
    results_df = pd.DataFrame({"k":k, "accuracy":accuracies, "fscore": fscores,
        "precision":precisions, "recall":recalls, "mean_absolute_error":maes,
        "mean_squared_error":mses,
        "root_mean_squared_error":rmses, "mean_absolute_error1":maes1,
        "mean_squared_error1":mses1,
        "root_mean_squared_error1":rmses1})
    results_df.to_csv(result_path+".csv", index=False)

    test, test1 = data_loader.prepare_data_next_activity(
        test_act_df,x_word_dict, y_word_dict,max_case_length)
    test2, test_time, test3, test4, __, ___, _____ = data_loader.prepare_data_times(
        test_time_df, x_word_dict, max_case_length, time_scaler, y_scaler, y_scaler1, False)
    y_p = transformer_model.predict([test,test2,test_time])
    #下一个事件预测结果
    y_p1 = np.argmax(y_p[0],axis=1)
    next_act_df = pd.DataFrame({"true value": test1, "predict value": y_p1})
    next_act_df.to_csv(next_act_path + ".csv", index=False)
    # 下一个事件时间预测结果
    y_p2 = y_p[1]
    test3 = y_scaler.inverse_transform(test3)
    y_p2 = y_scaler.inverse_transform(y_p2)
    test3 = test3.tolist()
    y_p2 = y_p2.tolist()
    next_time_df = pd.DataFrame({"true value": test3, "predict value": y_p2})
    next_time_df.to_csv(next_time_path + ".csv", index=False)
    # 剩余时间预测结果
    y_p3 = y_p[2]
    test4 = y_scaler.inverse_transform(test4)
    y_p3 = y_scaler.inverse_transform(y_p3)
    test4 = test4.tolist()
    y_p3 = y_p3.tolist()
    remaining_time_df = pd.DataFrame({"true value": test4, "predict value": y_p3})
    remaining_time_df.to_csv(remaining_time_path + ".csv", index=False)