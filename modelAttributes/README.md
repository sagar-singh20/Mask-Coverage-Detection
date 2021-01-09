# Mask Coverage Model 

Use this to directly load the trained model (.h5) and best weights based on highest validation accuracy. 


Script for Loading:

    maskDetector = load_model('modelAttributes/maskCoverageModel_v2.h5')
    maskDetector.load_weights("modelAttributes/weights_maskcoverage_v2.best.hdf5")



