// Loads the same vocab.json the Python side writes (word -> index), and
// decodes a sequence of ids back to a caption string the same way
// vision2words.data.vocabulary.Vocabulary.decode does: stop at <END>, skip
// <PAD>/<START>, fall back to <UNK> for an id with no matching word.
#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace v2w {

class Vocabulary {
public:
    static Vocabulary load(const std::string& path) {
        std::ifstream file(path);
        if (!file) {
            throw std::runtime_error("could not open vocab file: " + path);
        }
        nlohmann::json data;
        file >> data;

        Vocabulary vocab;
        for (auto& [word, index] : data.at("word2idx").items()) {
            int idx = index.get<int>();
            vocab.word_to_id_[word] = idx;
            vocab.id_to_word_[idx] = word;
        }
        vocab.pad_id_ = vocab.word_to_id_.at("<PAD>");
        vocab.start_id_ = vocab.word_to_id_.at("<START>");
        vocab.end_id_ = vocab.word_to_id_.at("<END>");
        vocab.unk_id_ = vocab.word_to_id_.at("<UNK>");
        return vocab;
    }

    int pad_id() const { return pad_id_; }
    int start_id() const { return start_id_; }
    int end_id() const { return end_id_; }
    int unk_id() const { return unk_id_; }
    size_t size() const { return word_to_id_.size(); }

    std::string decode(const std::vector<int64_t>& ids) const {
        std::string result;
        bool first = true;
        for (int64_t id : ids) {
            if (id == end_id_) break;
            if (id == pad_id_ || id == start_id_) continue;
            auto it = id_to_word_.find(static_cast<int>(id));
            const std::string& word = (it != id_to_word_.end()) ? it->second : std::string("<UNK>");
            if (!first) result += ' ';
            result += word;
            first = false;
        }
        return result;
    }

private:
    std::unordered_map<std::string, int> word_to_id_;
    std::unordered_map<int, std::string> id_to_word_;
    int pad_id_ = 0, start_id_ = 0, end_id_ = 0, unk_id_ = 0;
};

}  // namespace v2w
