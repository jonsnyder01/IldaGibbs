package org.knowceans.topics.simple;

import java.util.List;

final class WordData {
    private final int V;
    private final int[][] w;
    private final List<Integer> doc_list;

    public WordData(int V, int[][] w, List<Integer> doc_list) {
        this.V = V;
        this.w = w;
        this.doc_list = doc_list;
    }

    public int getV() {
        return V;
    }

    public int[][] getW() {
        return w;
    }

    public List<Integer> getDocList() {
        return doc_list;
    }
}
