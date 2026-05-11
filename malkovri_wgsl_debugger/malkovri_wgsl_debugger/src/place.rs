use naga::{Expression, GlobalVariable, Handle, LocalVariable};

use crate::value::Value;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PlaceRoot {
    Local {
        function_frame_index: usize,
        handle: Handle<LocalVariable>,
    },
    Global {
        handle: Handle<GlobalVariable>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PlaceSegment {
    Index(usize),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Place {
    pub(crate) root: PlaceRoot,
    pub(crate) path: Vec<PlaceSegment>,
}

impl Place {
    pub(crate) fn new(root: PlaceRoot) -> Self {
        Self {
            root,
            path: Vec::new(),
        }
    }

    pub(crate) fn with_index(mut self, index: usize) -> Self {
        self.path.push(PlaceSegment::Index(index));
        self
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ArgumentValue {
    Value(Value),
    Place(Place),
}

#[derive(Clone, Debug)]
pub(crate) enum EvaluatedExpression {
    Value(Value),
    Place(Place),
}

impl From<Value> for EvaluatedExpression {
    fn from(value: Value) -> Self {
        EvaluatedExpression::Value(value)
    }
}

impl From<Place> for EvaluatedExpression {
    fn from(place: Place) -> Self {
        EvaluatedExpression::Place(place)
    }
}

pub(crate) type ExpressionCache = std::collections::HashMap<Handle<Expression>, Value>;
