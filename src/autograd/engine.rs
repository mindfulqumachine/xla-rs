use super::GraphNode;
use std::collections::HashSet;
use std::rc::Rc;

/// Runs the backward pass starting from the given root node.
///
/// This function performs a topological sort of the computation graph to ensure
/// that dependencies are processed before their consumers. It then calls
/// `.backward()` on each node in reverse topological order.
pub fn backward(root: Option<Rc<dyn GraphNode>>) {
    let Some(root) = root else { return };

    let mut topo = Vec::new();
    let mut visited = HashSet::new();

    build_topo(root.clone(), &mut topo, &mut visited);

    for node in topo.into_iter().rev() {
        node.backward();
    }
}

/// Recursively builds the topological sort of the graph.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[derive(Debug)]
    struct MockNode {
        id: usize,
        parents: Vec<Rc<dyn GraphNode>>,
        visited_order: Rc<RefCell<Vec<usize>>>,
    }

    impl GraphNode for MockNode {
        fn backward(&self) {
            self.visited_order.borrow_mut().push(self.id);
        }

        fn parents(&self) -> Vec<Rc<dyn GraphNode>> {
            self.parents.clone()
        }
    }

    #[test]
    fn test_topological_sort() {
        let order = Rc::new(RefCell::new(Vec::new()));

        // Create a diamond graph:
        //   3
        //  / \
        // 1   2
        //  \ /
        //   0

        let n0 = Rc::new(MockNode {
            id: 0,
            parents: vec![],
            visited_order: order.clone(),
        });
        let n1 = Rc::new(MockNode {
            id: 1,
            parents: vec![n0.clone()],
            visited_order: order.clone(),
        });
        let n2 = Rc::new(MockNode {
            id: 2,
            parents: vec![n0.clone()],
            visited_order: order.clone(),
        });
        let n3 = Rc::new(MockNode {
            id: 3,
            parents: vec![n1.clone(), n2.clone()],
            visited_order: order.clone(),
        });

        backward(Some(n3));

        let result = order.borrow();
        // Expected order: 3 -> (1 or 2) -> (2 or 1) -> 0
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 3);
        assert_eq!(result[3], 0);
        assert!(result.contains(&1));
        assert!(result.contains(&2));
    }
}
