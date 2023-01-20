import pandas as pd
import matplotlib.pyplot as plt

def normalize(dataset):
    '''
        Normalize function will generate a normalized dataset
        ---
        Input: A dataset
        Output: A normalized dataset
    '''
    return ([(x - min(dataset)) / (max(dataset) - min(dataset)) for x in dataset])


def denormalize(thetas, dataset):
    '''
        Denormalize function will generate a denormalized theta
        ---
        Input: A theta
        Output: A denormalized theta
        ---
        delta is calculated by the max minus min for each column in the dataset
    '''
    delta = [max(value) - min(value) for value in (dataset['km'], dataset['price'])]
    thetas[1] = thetas[1] * delta[1] / delta[0] 
    thetas[0] = thetas[0] * delta[1] + min(dataset['price'])  - thetas[1] * min(dataset['km'])
    return thetas

def train(datas, learning_rate, theta):
    sum = [0, 0]
    for i in range(len(datas[0])):
        x, y = [row[i] for row in datas]
        estimated_y = theta[0] + (theta[1] * x)
        sum[0] += (estimated_y - y)
        sum[1] += (estimated_y - y) * x
    return [theta[i] - learning_rate * sum[i] / len(datas[i]) for i in range(len(theta))]

if __name__ == "__main__":
    learning_rate = 0.1
    theta = [0, 0]

    data = pd.read_csv('data.csv')

    price_norm = normalize(data['price'])
    km_norm = normalize(data['km'])

    plt.subplot(2, 2, 1)
    plt.scatter(data['km'], data['price'])
    plt.title('Before normalization')
    plt.subplot(2, 2, 2)
    plt.scatter(km_norm, price_norm)
    plt.title('After normalization')

    plt.subplot(2, 2, 3)
    plt.scatter(km_norm, price_norm)
    plt.title('Training')
    while True:
        new = train([km_norm, price_norm], learning_rate, theta)
        if new == theta:
            break
        theta = new
        plt.plot(km_norm, [theta[0] + (theta[1] * x) for x in km_norm], alpha=0.1)            
    denormalized_theta = denormalize(theta, data)

    plt.subplot(2, 2, 4)
    plt.title('Linear regression')
    plt.scatter(data['km'], data['price'])
    new_csv = pd.DataFrame({'theta0': [denormalized_theta[0]], 'theta1': [denormalized_theta[1]]})
    new_csv.to_csv('data_normalized.csv', index=False)

    plt.plot(data['km'], [denormalized_theta[0] + (denormalized_theta[1] * x) for x in data['km']], color='red')
    plt.show()