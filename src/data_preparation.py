from data.datamaker import DataProcessor

data_folder = '.../Acquisitions/Data/TVS'
data_format = 'data.mat'
folder_out = 'data/All'

def main():
    data_processor = DataProcessor(data_folder, data_format, folder_out)
    data_processor.process_data()
    
if __name__ == '__main__':
    main()


