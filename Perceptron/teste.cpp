if (line.empty()) continue;
        stringstream ss(line);
        string item;
        Vector features;
        for (int i = 0; i < 4; ++i) {
            getline(ss, item, ',');
            features.push_back(stod(item));
        }
        inputs.push_back(features);
        getline(ss, item);
        Vector target(3, 0.0);
        if (labelMap.find(item) != labelMap.end()) {
            target[labelMap[item]] = 1.0;
        }
        targets.push_back(target);