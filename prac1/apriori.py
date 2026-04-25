import numpy as np
import pandas as pd
#from pandas.api.types import CategoricalDtype

class ARApriori():
    def __init__(self, filepath ,sep=';', items_col='Items', items_sep=',', header='infer', encoding="utf-8"):
        # Создаем DataFrame из файла csv
        self.dataset_in = pd.read_csv(filepath, sep=sep, header=header, names=None, encoding=encoding)
        # Настройки вывода DataFrame
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        # Внутренний атрибут для хранения "формы" DataFrame
        self._ds_shape = self.dataset_in.shape
        self.dataset_final = pd.DataFrame({'Items': [], 'Support': []})  # DataFrame Для хранения конечного результата
        self.items_col = items_col      # Наименование колонки с наборами объектов/предметов
        self.items_sep = items_sep      # разделитель объектов/предметов в наборе

    def set_ds(self, dataset):
        self.dataset_in = dataset
        self._ds_shape = dataset.shape

    # вычисление поддержки (отношение количества записей, содержащих подмножество, к количесву всех записей в наборе данных)
    def _get_support(self, sup_cnt):
        return (sup_cnt*100)/self._ds_shape[0]

    # DataFrame - это изменяемый объект, т.о. при передаче в качестве аргумента ф-ии он не копируется!
    # Подсчет количества вхождений множества-кандидата в множества из набора данных
    def _get_itemset_cnt_iter(self, itemset, dataset):
        counter = 0
        for _, value in dataset.items():
            # Если множество кандидатов является подмножеством в текущем наборе, то увеличиваем счетчик
            if set(itemset).issubset(value):
                counter += 1
        return counter

    # подсчет количества вхождений множества-кандидата в множества из набора данных (метод apply)
    def _get_itemset_cnt_apply(self, itemset, dataset):
        return dataset.apply(lambda row: self._is_subset(itemset, row)).sum()

    # формирование списка кандидатов по порогу минимальной поддержки
    def _proc_candidates_set(self, candset, dataset, min_supp):
        print("Processing candidates...")
        df_buf = []
        # todo: можно оптимицировать цикл под объект Series (аргумент candset)
        for value in candset:
            # todo: Самая нагруженная часть кода!
            # Стандартная итерация
            supp = self._get_itemset_cnt_iter(value, dataset)
            # Метод apply
            #supp = self._get_itemset_cnt_apply(pd.Series(value), dataset)
            supp = self._get_support(supp)
            if supp > min_supp:
                df_buf.append({'Items': value, 'Support': supp})
        return pd.DataFrame(df_buf)

    # формирование начального списка кандидатов (по одному уникальному item в каждом наборе)
    # метод упрощен, т.к. не создает внутри себя отдельный Series, содержащий наборы item, т.к.
    # в качестве аргумента передается уже обработанный с помощью split объект Series (dataset)
    # todo: использовать оптимальные циклы
    def _get_items_ser2(self, itemset):
        items_list = []
        # Все наборы items из dataset добавляем в общий список
        for index, value in itemset.items():
            items_list += value
        # Конвертируем полученный список в объект Series и оставляем только уникальные значения (drop_duplicates)
        # todo: здесь можно выподнить преобразование уникальных строковых значений item в числовые
        result_ser = pd.Series(items_list, dtype=object).drop_duplicates().reset_index(drop=True)
        # Преобразуем каждый item в наборе в формат [item] (объект List) для возможности добавление новых item
        for index, value in result_ser.items():
            result_ser[index] = [value]
        return result_ser

    # Комбинируем items, которые прошли порог поддержки min_supp
    def _get_new_candidates(self, candset_old):
        print("Generating candidates...")
        candset_new = []    # список для хранения новых комбинаций items
        # в текущем списке множеств-кандидатов построчно просматриваем множетства ниже по таблице
        for index, val in candset_old.items():
            i = index + 1
            # если вышли за пределы таблицы, прерываем цикл
            if i >= candset_old.size:
                break
            # todo: здесь можно использовать рекуррентную функцию
            # иначе проверяем: содержится ли в текущем множестве (val) последний элемент из нижестоящего множества
            for _, subval in candset_old.iloc[i:].items():
                cand_new = list(val) # копируем список val
                sval = subval[-1]
                # если не содержится, то добавляем этот элемент к текущему множеству и получаем новое множество-кандидат
                if sval not in cand_new:
                    cand_new.append(subval[-1])
                    candset_new.append(cand_new)
        return pd.Series(candset_new).reset_index(drop=True)

    """ Вычисление support для всего набора входных данных
    min_supp - минимальный порог поддержки (support) """
    def get_ds_support(self, min_supp):
        # Получаем список, содержащий наборы данных из всех транзакций в виде набора списков
        items_set = self.dataset_in[self.items_col].str.split(self.items_sep)
        # Первый набор items для проверки.
        # Содержит списки, в которых находится по одному уникальному item
        candidates = self._get_items_ser2(items_set)
        print("First candidates:")
        print(candidates)
        K = 0       # Счетчик этапов генерации и обработки списков items
        while(True):
            K += 1
            # Вычисляем поддержку (support) для всех наборов items в списке кандидатов (candidates)
            # и фильтруем по пороговому значению min_supp
            validset = self._proc_candidates_set(candidates, items_set, min_supp)
            print("_____Step_%d_____" % K)
            print(validset, '\n')
            # если ни один из наборов items не прошел порог поддержки (min_supp), то завершаем цикл вычислений
            if validset.empty:
                break
            # иначе добавляем проверенное множество наборов items в итоговую таблицу данных (dataset_final)
            else:
                self.dataset_final = pd.concat([self.dataset_final, validset])
            # формируем новый список наборов-кандидатов (размером +1) из проверенного множества (validset)
            candidates = self._get_new_candidates(validset['Items'])

""" Конвертирование входных данных, имеющих вид:
T1 item1
T1 item2
T1 item3
...
в формат:
T1 item1,item2,item3,...
"""
def transform_in_dataset(dataset):
    col_id, col_item = dataset.columns.values
    ind = 0
    itemset = ""
    dataset_transformed = []
    dataset_transformed.append({ col_id: dataset.iloc[0][col_id], col_item: itemset})
    for index, row in dataset.iterrows():
        if row[col_id] == dataset_transformed[ind][col_id]:
            itemset += row[col_item] + ","
        else:
            dataset_transformed[ind][col_item] = itemset[:-1]
            ind += 1
            itemset = row[col_item] + ","
            dataset_transformed.append({ col_id: row[col_id], col_item: itemset})
    dataset_transformed[len(dataset_transformed)-1][col_item] = itemset[:-1]
    return pd.DataFrame(dataset_transformed)


if __name__ == '__main__':
    # Загрузка данных из csv-файла: sep - разделитель столбцов, items_sep - разделитель объектов в наборе,
    #                               items_col - имя колонки с наборами объектов
    apriori = ARApriori("checks.csv", sep=';', items_sep=',', items_col='ITEM')
    # Трансформация данных в формат T1 item1,item2,item3,... (если необходимо)
    ds_transformed = transform_in_dataset(apriori.dataset_in)
    apriori.set_ds(ds_transformed)
    # TEST
    #apriori = ARApriori("test_set.csv", sep=';', items_sep=',', items_col='Items')
    # Вычисляем support по заданному пороговому значению (в %)
    apriori.get_ds_support(1)
    print("_____RESULT_____")
    print(apriori.dataset_final, '\n')
    print("Save result to CSV ?:")
    q = input()
    if q.lower() == 'y':
        apriori.dataset_final.to_csv('result_supp.csv', index=True, sep=";", encoding="utf-8-sig")

