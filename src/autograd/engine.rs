use super::GraphNode;
use std::collections::HashSet;
use std::rc::Rc;

pub fn backward(root: Option<Rc<dyn GraphNode>>) {
    let Some(root) = root else { return };

    let mut topo = Vec::new();
    let mut visited = HashSet::new();

    // We need a way to identify nodes uniquely.
    // Rc pointers can be used for identity if we cast to *const ().
    // But `dyn GraphNode` is a fat pointer.
    // Let's assume the graph is a DAG and just traverse.
    // Actually, for topological sort we need visited set.
    // We can implement `id()` on GraphNode or use `Rc::as_ptr`.
    // `Rc::as_ptr` on trait object returns *const T (data pointer), which is stable.

    build_topo(root.clone(), &mut topo, &mut visited);

    for node in topo.into_iter().rev() {
        node.backward();
    }
}

fn build_topo(
    node: Rc<dyn GraphNode>,
    topo: &mut Vec<Rc<dyn GraphNode>>,
    visited: &mut HashSet<*const ()>,
) {
    let ptr = Rc::as_ptr(&node) as *const ();
    if visited.contains(&ptr) {
        return;
    }
    visited.insert(ptr);

    for parent in node.parents() {
        build_topo(parent, topo, visited);
    }

    topo.push(node);
}
