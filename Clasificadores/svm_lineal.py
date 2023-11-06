import numpy as np
from sklearn.svm import SVC as SVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as data_split

classifiers = {
	"SVM_linear" : SVM(kernel="linear"),
}
def svm_lineal(df):
    def classification(data_file, rounds=100, remove_disperse=[]):
        df_copy = data_file.copy() 
        
        if remove_disperse:
            df_copy = df_copy.drop(remove_disperse, axis=1)
        
        X = df_copy.drop('Label', axis=1)
        y = df_copy['Label']
        
        ans = {key: {"score" : []} for key, value in classifiers.items()}
        
        print("Classifying...")
        
        for i in range(rounds):
            X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.2)
            
            for name, classifier in classifiers.items():
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                
                classifier.fit(X_train, y_train)
                score = classifier.score(X_test, y_test)
                
                ans[name]["score"].append(score)
            
        print("Classification done!")
        
        return ans

    ans = classification(df)

    def sumary(ans, title="Summary"):
        size = 70
        separator = "-"
        
        print(separator*size)
        print("SUMARY: {}".format(title))
        print(separator*size)
        print("CLASSIF\t\tMEAN\tMEDIAN\tMINV\tMAXV\tSTD")
        print(separator*size)
        
        for n in ans:
            m = round(np.mean(ans[n]["score"])*100, 2)
            med = round(np.median(ans[n]["score"])*100, 2)
            minv = round(np.min(ans[n]["score"])*100, 2)
            maxv = round(np.max(ans[n]["score"])*100, 2)
            std = round(np.std(ans[n]["score"])*100, 2)
            
            print("{:<16}{}\t{}\t{}\t{}\t{}".format(n, m, med, minv, maxv, std))
        
        print(separator*size)
        print()

    sumary(ans)